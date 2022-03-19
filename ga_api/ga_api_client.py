import asyncio
from asyncore import loop
import os
import time
import base64
import getpass
import json
import math
import logging
from unicodedata import name
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes # 36.0.0
import aiohttp # 3.8.1
import pandas as pd # 1.4.2

def _logtime(ts):
    return time.strftime("%H:%M:%S", time.localtime(ts))

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
        logging.info(f'authenticate')
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

    async def _request(self, method, path, code, data=None):
        if time.time() - self._last_auth >= ACCESS_TOKEN_TIMEOUT: # need to authenticate
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
        retry_count = 0
        async with self._sem:
            while True:
                try:
                    # logging.debug(f'{method} {path}')
                    async with self._session.request(method, self.url+path, json=data, headers=self._headers, ssl=self.ssl) as resp:
                        if resp.status in code: # expected status code
                            if resp.status == 204:
                                rlt = None
                            else:
                                rlt = (await resp.json())['rlt']
                            return resp.status, rlt
                        # We can not retry here because Poll Dataset Data will return 202 if there
                        # are scheduled task. but we must retry here because if we release semaphone,
                        # we don't know when we can acquire it again.
                        elif resp.status == 202: # API server asks for retry
                            continue
                        else:
                            text = await resp.text()
                            logging.error(f'status={resp.status} message={text}')
                            raise UnexpectedStatus(resp.status, text)
                except aiohttp.ClientError as e:
                    if retry_count < self.retry: # user asks for retry
                        retry_count += 1
                        logging.warning(f'{method} {path}, retry {retry_count}')
                    else:
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
        _, rlt = await self._request('GET', '/cq/config', [200])
        return rlt

    async def set_system(self, config):
        """Set system config

        Args:
            config: The config part of system descriptor. See `System.get_system()`

        Returns:
            The modified system descriptor.
        """
        # 200 - ok, return modified system descriptor
        _, rlt = await self._request('PATCH', '/cq/config', [200], config)
        return rlt

    async def get_all_tenants(self):
        """Gets config and status of all tenants.

        Returns:
            Dictionary of tenant descriptors indexed by tenant id.
        """
        # 200 - ok, return array of tenant descriptor
        _, rlt = await self._request('GET', '/cq/config/tenant', [200])
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
        _, rlt = await self._request('POST', f'/cq/config/tenant/{tid}', [200], config)
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
        _, rlt = await self._request('PATCH', f'/cq/config/tenant/{tid}', [200], config)
        return rlt

    async def delete_tenant(self, tid):
        """Deletes tenant and its associated data.

        Args:
            tid: The target tenant id
        """
        # 204 - ok, return None
        await self._request('DELETE', f'/cq/config/tenant/{tid}', [204])

class Tenant(Client):
    def __init__(self, url, user, tenant, password, ssl=True, burst=1, retry=0):
        super().__init__(url, user, tenant, password, ssl=ssl, burst=burst, retry=retry)

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

    async def get_dataset(self, dsid):
        # 200 - ok, return read dataset descriptor
        status, rlt = await self._request('GET', f'/cq/dataset/{dsid}', [200])
        logging.info(f'status={status} dsid={dsid}')
        return rlt

    async def update_dataset(self, dsid, config):
        # 200 - ok, return modified dataset descriptor
        # 404 - dataset not exist
        status, rlt = await self._request('PATCH', f'/cq/dataset/{dsid}', [200], config)
        logging.info(f'status={status} dsid={dsid}')
        return rlt

    async def delete_dataset(self, dsid):
        # 204 - ok, return None
        status, rlt = await self._request('DELETE', f'/cq/dataset/{dsid}', [204])
        logging.info(f'status={status} dsid={dsid}')
        return rlt

    async def get_all_pipelines(self, dsid):
        # 200 - ok, return array of pipeline descriptors
        status, rlt = await self._request('GET', f'/cq/dataset/{dsid}/pipeline', [200])
        logging.info(f'status={status} dsid={dsid}')
        return rlt

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

    async def get_pipeline(self, dsid, plid):
        # 200 - ok, return read pipeline descriptor
        status, rlt = await self._request('GET', f'/cq/dataset/{dsid}/pipeline/{plid}', [200])
        logging.info(f'status={status} dsid={dsid} plid={plid}')
        return rlt

    async def update_pipeline(self, dsid, plid, config):
        # 200 - ok, return modified pipeline descriptor
        status, rlt = await self._request('PATCH', f'/cq/dataset/{dsid}/pipeline/{plid}', [200], config)
        logging.info(f'status={status} dsid={dsid} plid={plid}')
        return rlt

    async def delete_pipeline(self, dsid, plid):
        # 204 - ok, return None
        status, rlt = await self._request('DELETE', f'/cq/dataset/{dsid}/pipeline/{plid}', [204])
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

def _compute_ncol(pipeline):
    ncol = 0
    if 'bucket' in pipeline:
        for bkt in pipeline['bucket']:
            if bkt[0] in ['$distinctTuple','enumTuple']:
                # each field introduce one key column
                ncol += len(bkt[1]['fields'])
            else:
                # each tier introduce one key column
                ncol += 1
    if 'metric' in pipeline:
        # each metric introduce two columns
        ncol += len(pipeline['metric']) * 2
    return ncol

def _translate_table(input, ts, ncol):
    if isinstance(input, list):
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
    else: # None or Exception
        return []

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

    async def read_data(self, dts, series, columns=None, refresh=True):
        """Execute adhoc request and read its data

        Args:
            dts: pandas.DatetimeIndex or array-like.
            series: 'full', 'hour' or 'day'.
            columns: column labels of frame. Defaults to None.
            refresh: refresh adhoc on API server. Defaults to True.

        Returns:
            The adhoc data frame.
        """
        if refresh:
            await self._tenant.create_adhoc(self.pipeline)
        tasks = []
        tsidx = []
        freq_str = f"{self.freq}S"
        delta = pd.Timedelta(freq_str)
        for ts in dts:
            start = ts.floor(freq_str)
            end = start + delta
            coro = self._tenant.execute_adhoc(self.pid, start.timestamp(), end.timestamp(), series)
            tasks.append(asyncio.create_task(coro))
            tsidx.append(start)
        rlts = await asyncio.gather(*tasks, return_exceptions=True)
        data = []
        ncol = _compute_ncol(self.pipeline)
        for ts, rlt in zip(tsidx, rlts):
            tls = _translate_table(rlt, ts, ncol)
            data.extend(tls)
        return pd.DataFrame(data, columns=columns)

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

    async def read_data(self, dts, columns=None):
        """Read pipeline data on API server.

        Args:
            dts: pandas.DatetimeIndex or array-like.
            columns: Column labels of frame. Defaults to None.

        Returns:
            The pipeline data frame.
        """
        tasks = []
        tsidx = []
        freq_str = f"{self._dset.conf['freq']}S"
        for ts in dts:
            start = ts.floor(freq_str)
            coro = self._tenant.query_pipeline_data(self._dset.dsid, self.plid, start.timestamp())
            tasks.append(asyncio.create_task(coro))
            tsidx.append(start)
        rlts = await asyncio.gather(*tasks, return_exceptions=True)
        data = []
        ncol = _compute_ncol(self.conf['pipeline'])
        for ts, rlt in zip(tsidx, rlts):
            tls = _translate_table(rlt, ts, ncol)
            data.extend(tls)
        return pd.DataFrame(data, columns=columns)

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

    async def show_pipelines(self):
        """Get the config of all pipelines.

        Returns:
            dict of pipeline config indexed by pipeline id.
        """
        pipe_list = await self._tenant.get_all_pipelines(self.dsid)
        pipe_dict = {}
        for pipe in pipe_list:
            item = pipe['config'].copy()
            item.update(pipe['status'])
            del item['pipeline']
            pipe_dict[pipe['_id']] = item
        return pd.DataFrame.from_dict(pipe_dict, orient='index')

    async def del_all_pipelines(self):
        """Delete all pipelines of this dataset.
        """
        await self._tenant.delete_all_pipelines(self.dsid)

    async def get_pipeline(self, plid):
        """Get the specified pipeline.

        Args:
            plid (int): pipeline id

        Returns:
            The pipeline object with the specified id
        """
        pipe = await self._tenant.get_pipeline(self.dsid, plid)
        return Pipeline(self._tenant, self, plid, pipe['config'])

    async def set_pipeline(self, plid, config):
        """Open the specified pipeline. The pipeline is created if it does not
        exist.

        Args:
            plid: pipeline id
            config: pipeline config.

        Returns:
            The pipeline object
        """
        try:
            pipe = await self._tenant.update_pipeline(self.dsid, plid, config)
        except UnexpectedStatus as e:
            if e.status == 404:
                pipe = await self._tenant.create_pipeline(self.dsid, plid, config)
            else:
                raise e
        return Pipeline(self._tenant, self, plid, pipe['config'])

    async def del_pipeline(self, plid):
        """Delete the specified pipeline.

        Args:
            plid: pipeline id
        """
        await self._tenant.delete_pipeline(self.dsid, plid)

    async def patch_data(self, dts, overwrite=False):
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
            coro = self._tenant.patch_dataset_data(self.dsid, start.timestamp(), overwrite)
            tasks.append(asyncio.create_task(coro))
            tsidx.append(start)
        rlts = await asyncio.gather(*tasks, return_exceptions=True)
        rlts = [elem if isinstance(elem, dict) else {} for elem in rlts]
        return pd.DataFrame(rlts, index=pd.Index(dts, name='timestamp'))

    async def poll_data(self, dts):
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
            coro = self._tenant.poll_dataset_data(self.dsid, start.timestamp())
            tasks.append(asyncio.create_task(coro))
            tsidx.append(start)
        rlts = await asyncio.gather(*tasks, return_exceptions=True)
        rlts = [elem if isinstance(elem, dict) else {} for elem in rlts]
        return pd.DataFrame(rlts, index=pd.Index(tsidx, name='timestamp'))

    async def monitor_data(self, ts, coro):
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
        return asyncio.create_task(self._monitor_loop(next_ts, delta, coro))

    async def _monitor_loop(self, next_ts, delta, coro):
        while True:
            rlt = await self._tenant.poll_dataset_data(self.dsid, next_ts.timestamp())
            if rlt is None: # 204
                logging.debug(f'204 for {next_ts}, retried')
                await asyncio.sleep(3)
                continue
            await coro(next_ts)
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
        if user is None:
            user = input('User:')
        if tenant is None:
            tenant = input('Tenant:')
        if password is None:
            password = getpass.getpass('User Password:')
        self._tenant = Tenant(url, user, tenant, password, ssl=ssl, burst=burst, retry=retry)

    async def close(self):
        """Close the underlying connections gracefully. If the event loop is stopped before
        the repository is closed, a warning is emitted.
        """
        await self._tenant._session.close()
        await asyncio.sleep(0.250)

    async def show_datasets(self):
        """Get the configurations of all datasets.

        Returns:
            pandas.DataFrame: Dataset configurations indexed by dataset id.
        """
        dset_list = await self._tenant.get_all_datasets()
        dset_dict = {}
        for dset in dset_list:
            item = dset['config'].copy()
            item.update(dset['status'])
            dset_dict[dset['_id']] = item
        return pd.DataFrame.from_dict(dset_dict, orient='index')

    async def del_all_datasets(self):
        """Delete all datasets of this tenant.
        """
        await self._tenant.delete_all_datasets()

    async def get_dataset(self, dsid:int):
        """Get the specified dataset.

        Args:
            dsid: dataset id

        Returns:
            Dataset: The Dataset object
        """
        desc = await self._tenant.get_dataset(dsid)
        return Dataset(self._tenant, dsid, desc['config'])

    async def set_dataset(self, dsid:int, config:dict):
        """Update the specified dataset, or create it if not exist.

        Args:
            dsid: dataset id
            config: dataset config

        Returns:
            The created or updated dataset object
        """
        try:
            desc = await self._tenant.update_dataset(dsid, config)
        except UnexpectedStatus as e:
            if e.status == 404:
                desc = await self._tenant.create_dataset(dsid, config)
            else:
                raise e
        return Dataset(self._tenant, dsid, desc['config'])

    async def del_dataset(self, dsid:int):
        """Delete the specified dataset.

        Args:
            dsid: The dataset id
        """
        await self._tenant.delete_dataset(dsid)

    async def set_adhoc(self, freq:int, pipeline:dict):
        """Create an Adhoc Object.

        Args:
            freq: the aggregate frequency in seconds between 60 to 3600.
            pipeline: the pipeline

        Returns:
            Adhoc: The Adhoc object
        """
        return await Adhoc._create(self._tenant, freq, pipeline)