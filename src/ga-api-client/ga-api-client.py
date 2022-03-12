import asyncio
import aiohttp
import os
import time
import base64
import getpass
import json
import math
from functools import reduce
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

import pandas as pd
from collections import OrderedDict

def pprint(value, indent=4, depth=0):
    if isinstance(value, dict):
        # empty dict, printed as simple
        if len(value) == 0:
            print('{}', end='')
            return
        # non-empty dict, one item per line
        depth += indent
        print('{\n'+' '*depth,end='')
        for i, (k, v) in enumerate(value.items()):
            print(json.dumps(k), end=':')
            pprint(v, indent, depth)
            if i < len(value)-1: # not last item
                print(',\n' + ' '*depth, end='')
            else: # last item
                depth -= indent
                print('\n' + ' '*depth, end='')
        print('}', end='')
    elif isinstance(value, list):
        # empty list is printed as simple
        if len(value) == 0:
            print('[]', end='')
            return
        # non-empty list, check if expression
        any_complex = False
        if type(value[0]) != str or value[0][:1] != '$':
            for e in value:
                if (type(e) == dict and len(e) > 0) or \
                   (type(e) == list and len(e) > 0):
                    any_complex = True
                    break
        if any_complex:
            depth += indent
            print('[\n'+' '*depth, end='')
            for i, e in enumerate(value):
                pprint(e, indent, depth)
                if i < len(value)-1: # not last item
                    print(',\n' + ' '*depth, end='')
                else: # last item
                    depth -= indent
                    print('\n' + ' '*depth, end='')
            print(']', end='')
        else:
            print('[', end='')
            for i, e in enumerate(value):
                pprint(e, indent, depth)
                if i < len(value)-1: # not last item
                    print(',', end='')
            print(']', end='')
    else: # simple
        print(json.dumps(value), end='')

class HyperLogLog:
    def __init__(self, arg=None):
        if arg is None: # zero
            self.rmem = bytes(2048)
        elif isinstance(arg, str): # HyperLogLog Literal
            self.rmem = bytes.fromhex(arg[2:])
        elif type(arg) == bytes: # rmem
            self.rmem = arg
        else:
            raise Exception('HyperLogLog construct error')
        pass

    def __add__(self, other):
        return HyperLogLog(bytes(max(a,b) for a, b in zip(self.rmem, other.rmem)))

    def __str__(self):
        return '<HyperLogLog>'

    def value(self):
        N = 1 << 11
        V = 0
        S = 0
        for r in self.rmem:
            if r == 0:
                V += 1
            S += 2**(0-r)
        E = (N*N*0.7213/(1.0+1.079/N))/S
        if (E <= N*5.0/2.0 and V != 0): # small range correction
            E = N*math.log(N/V)
        elif (E >= (2**32)/30.0): # large range correction
            E = -1*(2**32)*math.log(1-E*(2**(-32)));
        return int(round(E))

AIO_TIMEOUT = 300
ACCESS_TOKEN_TIMEOUT = 86400 / 2

class ResponseError(Exception):
    def __init__(self, text='') -> None:
        super().__init__(text)

class Client:
    def __init__(self, url, user, tenant, password, ssl=True, burst=1, verbose=False):
        self.url = url # base url
        self.user = user # user name
        self.tenant = tenant # tenant name
        self.password = base64.b64encode(password.encode('utf-8')).decode() # base64 encoded password
        self.ssl = ssl
        self.sem = asyncio.Semaphore(burst)
        self.verbose = verbose
        self.session = aiohttp.ClientSession()
        self.headers = {
            'Content-Type': 'application/json'
        }
        self.last_auth = 0

    async def authenticate(self):
        if time.time() - self.last_auth < ACCESS_TOKEN_TIMEOUT:
            return
        authData = {
            'strategy': 'custom', # tells this is API request
            'account': f'{self.user}@{self.tenant}', # user account '<user>@<tenant>'
            'password': self.password # user password
        }
        # bringup
        if self.verbose:
            print(f'authenticate(): bringup')
        url = self.url+f'/bringup?name={self.tenant}'
        async with self.session.get(url, timeout=AIO_TIMEOUT, headers=self.headers, ssl=self.ssl) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f'status={resp.status}, text={text}')
            bringup = await resp.json()
        # do captcha when enabled
        if bringup['captchaEnabled']:
            if self.verbose:
                print(f'authenticate(): captcha')
            key = os.urandom(24)
            iv = os.urandom(16)
            data = {
                'key': key.hex(),
                'iv': iv.hex()
            }
            async with self.session.post(self.url+'/captcha', json=data, timeout=AIO_TIMEOUT, headers=self.headers, ssl=self.ssl) as resp:
                if resp.status != 201:
                    text = await resp.text()
                    raise Exception(f'status={resp.status}, text={text}')
                # decrypt answer
                captcha = await resp.json()
                ct = base64.b64decode(captcha['answer'])
                decryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).decryptor()
                answer_b = decryptor.update(ct) + decryptor.finalize()
                answer_s = answer_b.decode('utf8')
                authData['captcha'] = {
                    '_id': captcha['_id'],
                    'answer': answer_s
                }
        # do authenticate
        if self.verbose:
            print(f'authenticate(): authenticate')
        url = self.url+'/authentication'
        async with self.session.post(url, json=authData, timeout=AIO_TIMEOUT, headers=self.headers, ssl=self.ssl) as resp:
            if resp.status != 201:
                raise Exception(f'status={resp.status}, text={resp.text}')
            body = await resp.json()
            self.headers = {
                'Content-Type': 'application/json',
                'Authorization': body['accessToken']
            }
            self.last_auth = time.time()

    async def request(self, method, path, code, data=None):
        await self.authenticate()
        while True:
            async with self.sem:
                if self.verbose:
                    print(f'request(): {method} {path}')
                async with self.session.request(method, self.url+path, json=data, timeout=AIO_TIMEOUT, headers=self.headers, ssl=self.ssl) as resp:
                    if resp.status == code:
                        if method != 'DELETE':
                            return (await resp.json())['rlt']
                        else:
                            return None
                    elif resp.status == 202:
                        continue
                    else:
                        text = await resp.text()
                        raise Exception(f'status={resp.status}, text={text}')

class System(Client):
    def __init__(self, url, user, password, ssl=True, verbose=False):
        super().__init__(url, user, 'system', password, ssl=ssl, verbose=verbose)

    async def get_system(self):
        return await self.request('GET', '/cq/config', 200)

    async def set_system(self, config):
        return await self.request('PATCH', '/cq/config', 200, config)

    async def get_all_tenants(self):
        return await self.request('GET', '/cq/config/tenant', 200)

    async def get_tenant(self, tid):
        return await self.request('GET', f'/cq/config/tenant/{tid}', 200)

    async def create_tenant(self, tid, config):
        return await self.request('POST', f'/cq/config/tenant/{tid}', 200, config)

    async def update_tenant(self, tid, config):
        return await self.request('PATCH', f'/cq/config/tenant/{tid}', 200, config)

    async def delete_tenant(self, tid):
        return await self.request('DELETE', f'/cq/config/tenant/{tid}', 204)

class Tenant(Client):
    def __init__(self, url, user, tenant, password, ssl=True, burst=1, verbose=False):
        super().__init__(url, user, tenant, password, ssl=ssl, burst=burst, verbose=verbose)

    async def close(self):
        await self.session.close()

    async def create_adhoc(self, pipeline):
        return await self.request('POST', '/pipeline', 201, pipeline)

    async def execute_adhoc(self, pid, start, end, series):
        return await self.request('GET', f'/pipeline/{pid}?start={start}&end={end}&series={series}', 200)

    async def get_all_datasets(self):
        return await self.request('GET', '/cq/dataset', 200)

    async def delete_all_datasets(self):
        return await self.request('DELETE', '/cq/dataset', 204)

    async def create_dataset(self, dsid, config):
        return await self.request('POST', f'/cq/dataset/{dsid}', 200, config)

    async def get_dataset(self, dsid):
        return await self.request('GET', f'/cq/dataset/{dsid}', 200)

    async def update_dataset(self, dsid, config):
        return await self.request('PATCH', f'/cq/dataset/{dsid}', 200, config)

    async def delete_dataset(self, dsid):
        return await self.request('DELETE', f'/cq/dataset/{dsid}', 204)

    async def get_all_pipelines(self, dsid):
        return await self.request('GET', f'/cq/dataset/{dsid}/pipeline', 200)

    async def delete_all_pipelines(self, dsid):
        return await self.request('DELETE', f'/cq/dataset/{dsid}/pipeline', 204)

    async def create_pipeline(self, dsid, plid, config):
        return await self.request('POST', f'/cq/dataset/{dsid}/pipeline/{plid}', 200, config)

    async def get_pipeline(self, dsid, plid):
        return await self.request('GET', f'/cq/dataset/{dsid}/pipeline/{plid}', 200)

    async def update_pipeline(self, dsid, plid, config):
        return await self.request('PATCH', f'/cq/dataset/{dsid}/pipeline/{plid}', 200, config)

    async def delete_pipeline(self, dsid, plid):
        return await self.request('DELETE', f'/cq/dataset/{dsid}/pipeline/{plid}', 204)

    async def patch_dataset_data(self, dsid, ts, overwrite=False):
        # TODO: 204 will return error
        return await self.request('POST', f'/cq/dataset/{dsid}/task', 200, {
            'ts': ts,
            'overwrite': overwrite
        })

    async def poll_dataset_data(self, dsid, ts):
        # TODO: 204 will throw error
        return await self.request('POST', f'/cq/dataset/{dsid}/poll', 200, {
            'ts': ts
        })

    async def query_dataset_data(self, dsid, ts):
        return await self.request('GET', f'/cq/dataset/{dsid}/data?ts={ts}', 200)

    async def query_pipeline_data(self, dsid, plid, ts):
        return await self.request('GET', f'/cq/dataset/{dsid}/pipeline/{plid}/data?ts={ts}', 200)

def _aligh_date_range(ts, td, freq):
    freq_str = f"{freq}S"
    t1 = pd.Timestamp(ts).floor(freq_str)
    t2 = t1 + pd.Timedelta(freq_str if td is None else td)
    dr = pd.date_range(start=t1, end=t2, freq=freq_str, closed='left')
    return dr

def _translate_table(input, ts):
    output = []
    for row in input:
        _row = [ts]
        for elem in row:
            if type(elem) == list:
                _row.append(tuple(elem))
            elif type(elem) == str and elem.startswith('h:'):
                _row.append(HyperLogLog(elem))
            else:
                _row.append(elem)
        output.append(_row)
    return output

def _set_index(df, pipeline):
    nkey = 0
    if 'bucket' in pipeline:
        for bkt in pipeline['bucket']:
            if bkt[0] in ['$distinctTuple', '$enumTuple']:
                nkey += len(bkt[1]['fields'])
            else:
                nkey += 1
    return df.set_index(list(range(nkey+1)))

class Adhoc:
    def __init__(self, tenant, freq, pipeline):
        self.tenant = tenant # client to connect API server
        self.freq = freq # frequency in seconds
        self.pipeline = pipeline # pipeline object

    async def query_data(self, ts, td, series):
        dr = _aligh_date_range(ts, td, self.freq)
        pid = await self.tenant.create_adhoc(self.pipeline)
        tasks = []
        delta = pd.Timedelta(f'{self.freq}S')
        for ts in dr:
            start = ts.timestamp()
            end = (ts + delta).timestamp()
            coro = self.tenant.execute_adhoc(pid, start, end, series)
            tasks.append(asyncio.create_task(coro))
        rlts = await asyncio.gather(*tasks)
        data = []
        for ts, rlt in zip(dr, rlts):
            tls = _translate_table(rlt, ts)
            data.extend(tls)
        return _set_index(pd.DataFrame(data), self.pipeline)

class Pipeline:
    def __init__(self, tenant, dset, plid, conf):
        self.tenant = tenant
        self.dset = dset
        self.plid = plid
        self.conf = conf

    async def query_data(self, ts, td):
        dr = _aligh_date_range(ts, td, self.dset.conf['freq'])
        tasks = []
        for ts in dr:
            coro = self.tenant.query_pipeline_data(self.dset.dsid, self.plid, ts.timestamp())
            tasks.append(asyncio.create_task(coro))
        rlts = await asyncio.gather(*tasks)
        data = []
        for ts, rlt in zip(dr, rlts):
            data += _translate_table(rlt, ts)
        return _set_index(pd.DataFrame(data), self.conf['pipeline'])

class Dataset:
    def __init__(self, tenant, dsid, conf):
        self.tenant = tenant
        self.dsid = dsid
        self.conf = conf

    async def show_pipelines(self):
        pipe_list = await self.tenant.get_all_pipelines(self.dsid)
        pipe_dict = {}
        for pipe in pipe_list:
            item = pipe['config'].copy()
            item.update(pipe['status'])
            del item['pipeline']
            pipe_dict[pipe['_id']] = item
        return pd.DataFrame.from_dict(pipe_dict, orient='index')

    async def get_pipeline(self, plid):
        pipe = await self.tenant.get_pipeline(self.dsid, plid)
        return Pipeline(self.tenant, self, plid, pipe['config'])

    async def set_pipeline(self, plid, config):
        try:
            pipe = await self.tenant.create_pipeline(self.dsid, plid, config)
        except:
            pipe = await self.tenant.update_pipeline(self.dsid, plid, config)
        return Pipeline(self.tenant, self, plid, pipe['config'])

    async def del_pipeline(self, plid):
        await self.tenant.delete_pipeline(self.dsid, plid)

    async def del_all_pipelines(self, plid):
        await self.tenant.delete_all_pipelines(self.dsid)

    async def patch_data(self, ts, td, overwrite=False):
        dr = _aligh_date_range(ts, td, self.conf['freq'])
        tasks = []
        for ts in dr:
            coro = self.tenant.patch_dataset_data(self.dsid, ts.timestamp(), overwrite)
            tasks.append(asyncio.create_task(coro))
        rlts = await asyncio.gather(*tasks)
        return pd.DataFrame(rlts, index=dr)

    async def poll_data(self, ts, td):
        dr = _aligh_date_range(ts, td, self.conf['freq'])
        tasks = []
        for ts in dr:
            coro = self.tenant.poll_dataset_data(self.dsid, ts.timestamp())
            tasks.append(asyncio.create_task(coro))
        rlts = await asyncio.gather(*tasks)
        return pd.DataFrame(rlts, index=dr)

    async def query_data(self, ts, td):
        pipe_list = await self.tenant.get_all_pipelines(self.dsid)
        pipe_dict = {pipe['_id']: pipe['config']['pipeline'] for pipe in pipe_list}
        dr = _aligh_date_range(ts, td, self.conf['freq'])
        tasks = []
        for ts in dr:
            coro = self.tenant.query_dataset_data(self.dsid, ts.timestamp())
            tasks.append(asyncio.create_task(coro))
        rlts = await asyncio.gather(*tasks)
        data = {}
        for ts, rlt in zip(dr, rlts):
            for pkey, pdat in rlt.items():
                tls = _translate_table(pdat, ts)
                if pkey not in data:
                    data[pkey] = tls
                else:
                    data[pkey].extend(tls)
        dfd = {}
        for pkey, pdat in data.items():
            # the strange type casting is caused by using pid as object key
            df = _set_index(pd.DataFrame(pdat), pipe_dict[int(pkey)])
            dfd[pkey] = df
        return dfd

class Repository:
    def __init__(self, url, account=None, password=None, ssl=True, burst=1, verbose=False):
        if account is None:
            account = input('Account:')
        [user, tenant] = account.split('@')
        if password is None:
            password = getpass.getpass('Password:')
        self.tenant = Tenant(url, user, tenant, password, ssl=ssl, burst=burst, verbose=verbose)

    async def show_datasets(self):
        dset_list = await self.tenant.get_all_datasets()
        dset_dict = {}
        for dset in dset_list:
            item = dset['config'].copy()
            item.update(dset['status'])
            dset_dict[dset['_id']] = item
        return pd.DataFrame.from_dict(dset_dict, orient='index')

    async def get_dataset(self, dsid):
        desc = await self.tenant.get_dataset(dsid)
        return Dataset(self.tenant, dsid, desc['config'])

    async def set_dataset(self, dsid, config):
        try:
            desc = await self.tenant.create_dataset(dsid, config)
        except:
            desc = await self.tenant.update_dataset(dsid, config)
        return Dataset(self.tenant, dsid, desc['config'])

    async def del_dataset(self, dsid):
        await self.tenant.delete_dataset(dsid)

    async def set_adhoc(self, freq, pipeline):
        return Adhoc(self.tenant, freq, pipeline)
