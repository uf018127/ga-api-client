import asyncio
import itertools
import os
import time
import datetime
import base64
import getpass
import json
import logging
import math
from async_timeout import timeout
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes # 36.0.0
import aiohttp # 3.8.1
import pandas as pd # 1.4.2
from .hyper_log_log import HyperLogLog

def _logtime(ts):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))

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
    elif isinstance(value, str): # prevent escaped unicode
        print(f'"{value}"', end='')
    else: # simple
        print(json.dumps(value), end='')

class UnexpectedStatus(Exception):
    def __init__(self, status, message):
        self.status = status
        self.message = f'{status}: {message}'
        super().__init__(self.message)

ACCESS_TOKEN_TIMEOUT = 86400 / 2

class Client:
    def __init__(self, url, user, tenant, password, ssl=True, burst=1, retry=0):
        self.url = url # base url
        self.user = user # user name
        self.tenant = tenant # tenant name
        self.password = base64.b64encode(password.encode('utf-8')).decode() # base64 encoded password
        self.ssl = ssl # False to disable certificate verification
        self.retry = retry
        self._sem = asyncio.Semaphore(burst) # semaphone to throttle concurrent burst
        self._headers = {
            'Content-Type': 'application/json'
        }
        self._last_auth = 0 # timestamp of last authenticate in seconds since Eopch
        self._session = aiohttp.ClientSession()
        # timetout
        self.timeout = aiohttp.ClientTimeout(total=None, connect=None, sock_connect=30, sock_read=120)

    async def _authenticate(self):
        authData = {
            'strategy': 'custom', # tells this is API request
            'account': f'{self.user}@{self.tenant}', # user account '<user>@<tenant>'
            'password': self.password # user password
        }
        # bringup
        logging.info(f'bringup')
        url = self.url+f'/bringup?name={self.tenant}'
        async with self._session.get(url, headers=self._headers, ssl=self.ssl) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise UnexpectedStatus(resp.status, text)
            # TODO: if url is wrong, the result might not be JSON
            bringup = await resp.json()
        # do captcha when enabled
        if bringup['captchaEnabled']:
            logging.info(f'captcha')
            key = os.urandom(24)
            iv = os.urandom(16)
            data = {
                'key': key.hex(),
                'iv': iv.hex()
            }
            async with self._session.post(self.url+'/captcha', json=data, headers=self._headers, ssl=self.ssl) as resp:
                if resp.status != 201:
                    text = await resp.text()
                    raise UnexpectedStatus(resp.status, text)
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
        url = self.url+'/authentication'
        async with self._session.post(url, json=authData, headers=self._headers, ssl=self.ssl) as resp:
            if resp.status != 201:
                text = await resp.text()
                raise UnexpectedStatus(resp.status, text)
            body = await resp.json()
            self._headers = {
                'Content-Type': 'application/json',
                'Authorization': body['accessToken']
            }
            self._last_auth = time.time()
            logging.info(f'authenticate done')

    async def _request(self, method, path, code, data=None):
        # authenticate if required
        if time.time() - self._last_auth >= ACCESS_TOKEN_TIMEOUT:
            retry_count = 0
            while True:
                try:
                    await self._authenticate()
                    break
                except aiohttp.ClientError as e:
                    if retry_count < self.retry: # user asks for retry
                        retry_count += 1
                        logging.warning(f'retry {retry_count}-th time')
                    else:
                        raise e
        # send the request
        retry_count = 0
        async with self._sem:
            while True:
                try:
                    # logging.debug(f'{method} {path}')
                    async with self._session.request(method, self.url+path, json=data, headers=self._headers, ssl=self.ssl, timeout=self.timeout) as resp:
                        if resp.status in code: # expected status code
                            rlt = None
                            if resp.content_type == 'application/json':
                                data = await resp.json()
                                if 'rlt' in data:
                                    rlt = data['rlt']
                            return resp.status, rlt
                        elif resp.status == 202: # API server asks for retry
                            text = await resp.text()
                            logging.debug(f'status={resp.status} message={text}')
                            continue
                        else:
                            text = await resp.text()
                            logging.error(f'unexpected status={resp.status} message={text}')
                            raise UnexpectedStatus(resp.status, text)
                except aiohttp.ClientError as e:
                    if retry_count < self.retry: # user asks for retry
                        retry_count += 1
                        logging.warning(f'{method} {path}, retry {retry_count}')
                    else:
                        logging.error(f'{method} {path} fail')
                        raise e

class System(Client):
    def __init__(self, url, user=None, password=None, ssl=True, retry=0):
        """Construct object to send system requests.

        Args:
            url: The base url of API server.
            user: The username.
            password: The password.
            ssl: `False` to relax certificate verification.
            retry: Number of retry at network failure.
        """
        if user is None:
            user = input('User:')
        if password is None:
            password = getpass.getpass('Password:')
        super().__init__(url, user, 'system', password, ssl=ssl, retry=retry)

    async def close(self):
        """Close the underlying connections gracefully. If the event loop is stopped before
        the system object is closed, a warning is emitted.
        """
        await self._session.close()
        await asyncio.sleep(0.250)

    async def get_system(self):
        """Get system config and status

        Returns:
            System descriptor. For example::

            {
                "config":{
                    "burst":1,
                    "retry":1,
                    "maxTasks":65536
                },
                "status":{
                    "created":"2022-02-17T10:18:06.714Z",
                    "modified":"2022-03-14T17:44:46.717Z",
                    "size":38,
                    "maxId":36
                }
            }
        """
        # 200 - ok, return read system descriptor
        status, rlt = await self._request('GET', '/cq/config', [200])
        logging.info(f'status={status}')
        return rlt

    async def set_system(self, config):
        """Set system config

        Args:
            config: The config part of system descriptor. See `System.get_system()`

        Returns:
            The modified system descriptor.
        """
        # 200 - ok, return modified system descriptor
        status, rlt = await self._request('PATCH', '/cq/config', [200], config)
        logging.info(f'status={status}')
        return rlt

    async def get_all_tenants(self):
        """Gets config and status of all tenants.

        Returns:
            Dictionary of tenant descriptors indexed by tenant id.
        """
        # 200 - ok, return array of tenant descriptor
        status, rlt = await self._request('GET', '/cq/config/tenant', [200])
        logging.info(f'status={status}')
        return rlt

    async def get_tenant(self, tid):
        """Get tenant config and status.

        Args:
            tid: The target tenant id

        Returns:
            Descriptor of the target tenant. For example::

            {
                "_id":0,
                "config":{
                    "limit":10000000000
                },
                "status":{
                    "created":"2022-03-14T17:46:28.860Z",
                    "modified":"2022-03-14T18:31:51.716Z",
                    "size":0
                }
            }
        """
        # 200 - ok, return read tenant descriptor
        status, rlt = await self._request('GET', f'/cq/config/tenant/{tid}', [200])
        logging.info(f'status={status}')
        return rlt if status == 200 else None

    async def create_tenant(self, tid, config):
        """Creates a new tenant

        Args:
            tid: The target tenant id
            config: The config part of tenant descriptor. See `System.get_tenant()`

        Returns:
            Descriptor of the created tenant
        """
        # 200 - ok, return created tenant descriptor
        status, rlt = await self._request('POST', f'/cq/config/tenant/{tid}', [200], config)
        logging.info(f'status={status} tid={tid}')
        return rlt

    async def update_tenant(self, tid, config):
        """Updates tenant config

        Args:
            tid: The target tenant id
            config: The config part of tenant descriptor. See `System.get_tenant()`

        Returns:
            Descriptor of the target tenant
        """
        # 200 - ok, return modified tenant descriptor
        status, rlt = await self._request('PATCH', f'/cq/config/tenant/{tid}', [200], config)
        logging.info(f'status={status} tid={tid}')
        return rlt

    async def delete_tenant(self, tid):
        """Deletes tenant and its associated data.

        Args:
            tid: The target tenant id
        """
        # 204 - ok, return None
        status, rlt = await self._request('DELETE', f'/cq/config/tenant/{tid}', [204])
        logging.info(f'status={status} tid={tid}')

def desc_dataframe(desc_list):
    desc_dict = {}
    for dset in desc_list:
        item = dset['config'].copy()
        item.update(dset['status'])
        item.pop('pipeline', None)
        desc_dict[dset['_id']] = item
    return pd.DataFrame.from_dict(desc_dict, orient='index')

class Tenant(Client):
    def __init__(self, url, user=None, tenant=None, password=None, ssl=True, burst=1, retry=0):
        if user is None:
            user = input('User:')
        if tenant is None:
            tenant = input('Tenant:')
        if password is None:
            password = getpass.getpass('Password:')
        super().__init__(url, user, tenant, password, ssl=ssl, burst=burst, retry=retry)

    async def close(self):
        """Close the underlying connections gracefully. If the event loop is stopped before
        the tenant object is closed, a warning is emitted.
        """
        await self._session.close()
        await asyncio.sleep(0.250)

    async def create_adhoc(self, pipeline):
        # 201 - ok, return adhoc id
        status, rlt = await self._request('POST', '/pipeline', [201], pipeline)
        logging.info(f'status={status} rlt={rlt}')
        return rlt

    async def execute_adhoc(self, pid, start, end, series):
        # 200 - ok, return tabular data
        # 202 - accepted, try again to get data
        status, rlt = await self._request('GET', f'/pipeline/{pid}?start={start}&end={end}&series={series}', [200])
        logging.info(f'status={status} pid={pid} start={_logtime(start)} end={_logtime(end)} series={series}')
        return rlt if isinstance(rlt, list) else []

    async def get_all_datasets(self):
        # 200 - ok, return array of dataset descriptor
        status, rlt = await self._request('GET', '/cq/dataset', [200])
        logging.info(f'status={status} rlt={rlt}')
        return rlt

    async def list_all_datasets(self):
        dset_list = await self.get_all_datasets()
        return desc_dataframe(dset_list)

    async def delete_all_datasets(self):
        # 204 - ok, return None
        status, rlt = await self._request('DELETE', '/cq/dataset', [204])
        logging.info(f'status={status} rlt={rlt}')
        return rlt

    async def create_dataset(self, dsid, config):
        # 200 - ok, return created dataset descriptor
        status, rlt = await self._request('POST', f'/cq/dataset/{dsid}', [200], config)
        logging.info(f'status={status} dsid={dsid}')
        return rlt

    async def get_dataset(self, dsid, missing_ok=False):
        # 200 - ok, return read dataset descriptor
        # 404 - error, dataset not found
        code = [200, 404] if missing_ok else [200]
        status, rlt = await self._request('GET', f'/cq/dataset/{dsid}', code)
        if status != 404 or not missing_ok:
            logging.info(f'status={status} dsid={dsid}')
            return rlt
        else:
            return None

    async def update_dataset(self, dsid, config, missing_ok=False):
        # 200 - ok, return modified dataset descriptor
        # 404 - error, dataset not found
        code = [200, 404] if missing_ok else [200]
        status, rlt = await self._request('PATCH', f'/cq/dataset/{dsid}', code, config)
        if status != 404 or not missing_ok:
            logging.info(f'status={status} dsid={dsid}')
            return rlt
        else:
            return None

    async def delete_dataset(self, dsid, missing_ok=False):
        # 204 - ok, return None
        # 404 - error, dataset not found
        code = [204, 404] if missing_ok else [204]
        status, rlt = await self._request('DELETE', f'/cq/dataset/{dsid}', code)
        if status != 404 or not missing_ok:
            logging.info(f'status={status} dsid={dsid}')
        return rlt

    async def get_all_pipelines(self, dsid):
        # 200 - ok, return array of pipeline descriptors
        status, rlt = await self._request('GET', f'/cq/dataset/{dsid}/pipeline', [200])
        logging.info(f'status={status} dsid={dsid}')
        return rlt

    async def list_all_pipelines(self, dsid):
        pipe_list = await self.get_all_pipelines(dsid)
        return desc_dataframe(pipe_list)

    async def delete_all_pipelines(self, dsid):
        # 204 - ok, return None
        status, rlt = await self._request('DELETE', f'/cq/dataset/{dsid}/pipeline', [204])
        logging.info(f'status={status} dsid={dsid}')
        return rlt

    async def create_pipeline(self, dsid, plid, config):
        # 200 - ok, return created pipeline descriptor
        status, rlt = await self._request('POST', f'/cq/dataset/{dsid}/pipeline/{plid}', [200], config)
        logging.info(f'status={status} dsid={dsid} plid={plid}')
        return rlt

    async def get_pipeline(self, dsid, plid, missing_ok=False):
        # 200 - ok, return read pipeline descriptor
        # 404 - error, pipeline not found
        code = [200, 404] if missing_ok else [200]
        status, rlt = await self._request('GET', f'/cq/dataset/{dsid}/pipeline/{plid}', code)
        if status != 404 or not missing_ok:
            logging.info(f'status={status} dsid={dsid} plid={plid}')
            return rlt
        else:
            return None

    async def update_pipeline(self, dsid, plid, config, missing_ok=False):
        # 200 - ok, return modified pipeline descriptor
        # 404 - error, dataset not found
        code = [200, 404] if missing_ok else [200]
        status, rlt = await self._request('PATCH', f'/cq/dataset/{dsid}/pipeline/{plid}', code, config)
        if status != 404 or not missing_ok:
            logging.info(f'status={status} dsid={dsid} plid={plid}')
            return rlt
        else:
            return None

    async def delete_pipeline(self, dsid, plid, missing_ok=False):
        # 204 - ok, return None
        # 404 - error, pipeline not found
        code = [204, 404] if missing_ok else [204]
        status, rlt = await self._request('DELETE', f'/cq/dataset/{dsid}/pipeline/{plid}', code)
        if status != 404 or not missing_ok:
            logging.info(f'status={status} dsid={dsid} plid={plid}')
        return rlt

    async def patch_dataset_data(self, dsid, ts, overwrite=False):
        # 200 - ok, return dict of patch result (1:data, 0:no data)
        # 202 - accepted, try again to get data
        # 204 - future, return None
        # 400 - purged, return ??
        status, rlt = await self._request('POST', f'/cq/dataset/{dsid}/task', [200, 204], {
            'ts': ts,
            'overwrite': overwrite
        })
        logging.info(f'status={status} dsid={dsid} ts={_logtime(ts)}')
        return None if status != 200 else rlt

    async def poll_dataset_data(self, dsid, ts):
        # 200 - ok, return [<row>,...]
        # 202 - scheduled task
        # 204 - future, return None
        status, rlt = await self._request('POST', f'/cq/dataset/{dsid}/poll', [200, 204], {
            'ts': ts
        })
        logging.info(f'status={status} dsid={dsid} ts={_logtime(ts)}')
        return None if status != 200 else rlt

    async def query_dataset_data(self, dsid, ts):
        # 200 - ok, return dictionary of [<row>,...], key is str(plid)
        _, rlt = await self._request('GET', f'/cq/dataset/{dsid}/data?ts={ts}', [200])
        return rlt if isinstance(rlt, dict) else {}

    async def query_pipeline_data(self, dsid, plid, ts):
        # 200 - ok, return [<row>,...]
        status, rlt = await self._request('GET', f'/cq/dataset/{dsid}/pipeline/{plid}/data?ts={ts}', [200])
        logging.info(f'status={status} dsid={dsid} plid={plid} ts={_logtime(ts)}')
        return rlt if isinstance(rlt, list) else []

def _compute_kcol_mcol(pipeline):
    kcol = 0 # key columns
    mcol = 0 # mtr columns
    if 'bucket' in pipeline:
        for bkt in pipeline['bucket']:
            if bkt[0] in ['$distinctTuple','enumTuple']:
                # each field introduce one key column
                kcol += len(bkt[1]['fields'])
            else:
                # each tier introduce one key column
                kcol += 1
    if 'metric' in pipeline:
        # each metric introduce two columns
        mcol += len(pipeline['metric']) * 2
    return kcol, mcol

class _translate_table:
    def __init__(self, table, ts, kcol, mcol):
        if isinstance(table, list) and len(table) > 0:
            self.table = table
        else:
            key_cols = list(itertools.repeat('!all', kcol))
            mtr_cols = list(itertools.repeat(math.nan, mcol))
            self.table = [key_cols+mtr_cols]
        self.ts = ts

    def __iter__(self):
        for input_row in self.table:
            output_row = [] if self.ts is None else [self.ts]
            for elem in input_row:
                if type(elem) == list:
                    output_row.append(tuple(elem))
                elif type(elem) == str and elem.startswith('h:'):
                    output_row.append(HyperLogLog(elem))
                else:
                    output_row.append(elem)
            yield output_row

class _translate_rlts:
    def __init__(self, tsidx, rlts, kcol, mcol):
        self.tsidx = tsidx
        self.rlts = rlts
        self.kcol = kcol
        self.mcol = mcol

    def __iter__(self):
        for ts, rlt in zip(self.tsidx, self.rlts):
            yield _translate_table(rlt, ts, self.kcol, self.mcol)

TZINFO = datetime.datetime.now().astimezone().tzinfo
def _get_utc_timestamp(ts):
    if ts.tz is None:
        ts = ts.tz_localize(tz=TZINFO)
    return ts.timestamp()

class Adhoc:
    @staticmethod
    async def _create(tenant, freq, pipeline):
        """Create Adhoc object to send adhoc request

        Args:
            tenant: tenant object to access API server
            freq: frequency in seconds
            pipeline: pipeline object

        Returns:
            The created Adhoc object.
        """
        self = Adhoc()
        self._tenant = tenant
        self.freq = freq
        self.pipeline = pipeline # pipeline object
        self.pid = await self._tenant.create_adhoc(self.pipeline)
        return self

    async def _read_point(self, ts, series, columns=None, refresh=True):
        if refresh:
            await self._tenant.create_adhoc(self.pipeline)
        freq_str = f"{self.freq}S"
        delta = pd.Timedelta(freq_str)
        start = pd.Timestamp(ts).floor(freq_str)
        end = start + delta
        table = await self._tenant.execute_adhoc(self.pid, _get_utc_timestamp(start), _get_utc_timestamp(end), series)
        kcol, mcol = _compute_kcol_mcol(self.pipeline)
        return pd.DataFrame(_translate_table(table, None, kcol, mcol), columns=columns)

    async def _read_range(self, dts, series, columns=None, refresh=True):
        if refresh:
            await self._tenant.create_adhoc(self.pipeline)
        tasks = []
        tsidx = []
        freq_str = f"{self.freq}S"
        delta = pd.Timedelta(freq_str)
        for ts in dts:
            start = ts.floor(freq_str)
            end = start + delta
            coro = self._tenant.execute_adhoc(self.pid, _get_utc_timestamp(start), _get_utc_timestamp(end), series)
            tasks.append(asyncio.create_task(coro))
            tsidx.append(start)
        rlts = await asyncio.gather(*tasks, return_exceptions=True)
        kcol, mcol = _compute_kcol_mcol(self.pipeline)
        iter = itertools.chain.from_iterable(_translate_rlts(tsidx, rlts, kcol, mcol))
        return pd.DataFrame(iter, columns=columns)

    async def read_data(self, dts, *args, **kwargs):
        """Execute adhoc request and read its data

        Args:
            dts: str, pandas.Timestamp or array-like.
            series: 'full', 'hour' or 'day'.
            columns: column labels of frame. Defaults to None.
            refresh: refresh adhoc on API server. Defaults to True.

        Returns:
            The adhoc data frame.
        """
        try:
            ts = pd.Timestamp(dts)
        except:
            ts = None
        if ts is None:
            return await self._read_range(dts, *args, **kwargs)
        else:
            return await self._read_point(ts, *args, **kwargs)

class Pipeline:
    def __init__(self, tenant, dset, plid, conf):
        """Construct pipeline object to send pipeline requests.

        Args:
            tenant: tenant object to access API server
            dset: parent dataset object.
            plid: pipeline ID
            conf: pipeline config
        """
        self._tenant = tenant
        self._dset = dset
        self.plid = plid
        self.conf = conf

    async def _read_point(self, ts, columns=None):
        freq_str = f"{self._dset.conf['freq']}S"
        start = ts.floor(freq_str)
        table = await self._tenant.query_pipeline_data(self._dset.dsid, self.plid, _get_utc_timestamp(start))
        kcol, mcol = _compute_kcol_mcol(self.conf['pipeline'])
        return pd.DataFrame(_translate_table(table, None, kcol, mcol), columns=columns)

    async def _read_range(self, dts, columns=None):
        tasks = []
        tsidx = []
        freq_str = f"{self._dset.conf['freq']}S"
        for ts in dts:
            start = ts.floor(freq_str)
            coro = self._tenant.query_pipeline_data(self._dset.dsid, self.plid, _get_utc_timestamp(start))
            tasks.append(asyncio.create_task(coro))
            tsidx.append(start)
        rlts = await asyncio.gather(*tasks, return_exceptions=True)
        kcol, mcol = _compute_kcol_mcol(self.conf['pipeline'])
        iter = itertools.chain.from_iterable(_translate_rlts(tsidx, rlts, kcol, mcol))
        return pd.DataFrame(iter, columns=columns)

    async def read_data(self, dts, *args, **kwargs):
        """Read pipeline data on API server.

        Args:
            dts: pandas.Timestamp or array-like.
            columns: Column labels of frame. Defaults to None.

        Returns:
            The pipeline data frame.
        """
        try:
            ts = pd.Timestamp(dts)
        except:
            ts = None
        if ts is None:
            return await self._read_range(dts, *args, **kwargs)
        else:
            return await self._read_point(ts, *args, **kwargs)

class Dataset:
    def __init__(self, tenant, dsid, conf):
        """Construct dataset object to send dataset requests. Don't call directly.

        Args:
            tenant: tenant object to access API server
            dsid: dataset id
            conf: dataset config
        """
        self._tenant = tenant
        self.dsid = dsid
        self.conf = conf

    async def list(self):
        """List all pipelines

        Returns:
            pandas.DataFrame: Pipeline configurations indexed by id.
        """
        pipe_list = await self._tenant.get_all_pipelines(self.dsid)
        return desc_dataframe(pipe_list)

    async def pipeline(self, plid):
        """Get pipeline

        Returns:
            dset: Pipeline object with the specified id.
        """
        desc = await self._tenant.get_pipeline(self.dsid, plid)
        return Pipeline(self._tenant, self, plid, desc['config'])

    async def patch(self, dts, overwrite=False):
        """Patch dataset data

        Args:
            dts: pandas.DatetimeIndex or array-like.
            overwrite: True to force overwrite data on API server. Defaults to False.

        Returns:
            Patch result in frame.
        """
        freq_str = f"{self.conf['freq']}S"
        tasks = []
        tsidx = []
        for ts in dts:
            start = ts.floor(freq_str)
            coro = self._tenant.patch_dataset_data(self.dsid, _get_utc_timestamp(start), overwrite)
            tasks.append(asyncio.create_task(coro))
            tsidx.append(start)
        rlts = await asyncio.gather(*tasks, return_exceptions=True)
        rlts = [elem if isinstance(elem, dict) else {} for elem in rlts]
        return pd.DataFrame(rlts, index=pd.Index(dts, name='timestamp'))

    async def poll(self, dts):
        """Check dataset data availability.

        Args:
            dts: pandas.DatetimeIndex or array-like.

        Returns:
            Patch result in frame.
        """
        freq_str = f"{self.conf['freq']}S"
        tasks = []
        tsidx = []
        for ts in dts:
            start = ts.floor(freq_str)
            coro = self._tenant.poll_dataset_data(self.dsid, _get_utc_timestamp(start))
            tasks.append(asyncio.create_task(coro))
            tsidx.append(start)
        rlts = await asyncio.gather(*tasks, return_exceptions=True)
        rlts = [elem if isinstance(elem, dict) else {} for elem in rlts]
        return pd.DataFrame(rlts, index=pd.Index(tsidx, name='timestamp'))

    async def monitor(self, ts, coro, *args):
        """Create a task to monitor dataset data.

        Args:
            ts: start time in pandas.Timestamp
            coro: coroutine to handle new data

        Returns:
            The created task
        """
        freq_str = f"{self.conf['freq']}S"
        next_ts = pd.Timestamp(ts).floor(freq_str)
        delta = pd.Timedelta(freq_str)
        return asyncio.create_task(self._monitor_loop(next_ts, delta, coro, *args))

    async def _monitor_loop(self, next_ts, delta, coro, *args):
        while True:
            rlt = await self._tenant.poll_dataset_data(self.dsid, _get_utc_timestamp(next_ts))
            if rlt is None: # 204
                logging.debug(f'204 for {next_ts}, retried')
                await asyncio.sleep(3)
                continue
            await coro(next_ts, *args)
            next_ts += delta

class Repository:
    def __init__(self, url, user=None, tenant=None, password=None, ssl=True, burst=1, retry=0):
        """Construct repository object to send tenant requests to API server

        Args:
            url: The base URL of API server.
            user: The user name.
            tenant: The tenant name.
            password: The user password.
            ssl: Set False to relax certification checks for self signed API server. Defaults to True.
            burst: Maximum concurrent requests to the API server.
            retry: Number of retry at fail of sending request.
        """
        self._tenant = Tenant(url, user, tenant, password, ssl=ssl, burst=burst, retry=retry)

    async def close(self):
        """Close the underlying connections gracefully. If the event loop is stopped before
        the repository is closed, a warning is emitted.
        """
        await self._tenant.close()
        await asyncio.sleep(0.250)

    async def list(self):
        """List all datasets

        Returns:
            pandas.DataFrame: Dataset configurations indexed by id.
        """
        dset_list = await self._tenant.get_all_datasets()
        return desc_dataframe(dset_list)

    async def dataset(self, dsid):
        """Get dataset

        Returns:
            dset: Dataset object with the specified id.
        """
        desc = await self._tenant.get_dataset(dsid)
        return Dataset(self._tenant, dsid, desc['config'])

    async def adhoc(self, freq, pipeline):
        """Create an Adhoc Object.

        Args:
            freq: the aggregate frequency in seconds between 60 to 3600.
            pipeline: the pipeline

        Returns:
            Adhoc: The Adhoc object
        """
        return await Adhoc._create(self._tenant, freq, pipeline)
