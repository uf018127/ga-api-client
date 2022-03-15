from src.ga_api_package.ga_api_client import Repository, Tenant
import asyncio
import logging
import pandas as pd

async def open_main():
    # return Repository('https://rdlab-214.genie-analytics.com/api', 'api@default', 'api123!@#', burst=1)
    return Repository('http://192.168.11.214:3030/api', 'api@default', 'api123!@#', ssl=False, burst=1)

async def exec_main(repo):
    async def tick(ts):
        nonlocal df
        df_input = await pipe.query_data(ts, '5T')
        if len(df_input):
            df_input.columns = columns
            threshold = df.quantile(0.95)
            df = pd.concat([df.iloc[1:], df_input], ignore_index=True)
            df_now = pd.Series(df_input.iloc[0])
            df_now['f_ratio'] = df_now['forward'] / threshold['forward']
            df_now['o_ratio'] = df_now['opposite'] / threshold['opposite']
            if ((df_now['f_ratio'] > 2.0) & (df_now['o_ratio'] > 2.0)):
                logging.warning(df_now.to_json(date_format='iso'))

    columns = ['timestamp','forward','opposite']

    # create or update dataset
    dset = await repo.set_dataset(13579, {
        'name':'example2',
        'series':'hour',
        'freq':300,
        'retainDepth':86400*7,
        'retainSize':10000000,
        'run':True
    })

    # create or update pipeline
    pipe = await dset.set_pipeline(1001, {
        'name':'example',
        'pipeline': {
            'scope': {
                'i-field': '@controller.home',
                'i-entry': [
                    [True,'t:0:0:1',0]
                ],
                'mode': '%directional'
            },
            'metric': [
                ['$sum', {
                    'field':'@flow.bytes'
                }]
            ]
        }
    })

    # patch previous 48 hours data
    end = pd.Timestamp.now('+08:00').floor('5T') -  pd.Timedelta('5T')
    start = end - pd.Timedelta('48H')
    await dset.patch_data(start, '48H')

    # monitor latest
    end = start + pd.Timedelta('24H')
    df = await pipe.query_data(start, '24H')
    df.columns = columns
    task = await dset.monitor_data(end, tick)
    await asyncio.gather(task);

async def close_main(repo):
    await repo.close()
    await asyncio.sleep(0.250)

if __name__ == '__main__':
    logging.basicConfig(format='▸ %(asctime)s %(filename)s:%(lineno)d %(levelname)s %(message)s', level=logging.INFO)
    loop = asyncio.get_event_loop()
    logging.info('Running the loop')
    try:
        repo = loop.run_until_complete(open_main())
        loop.run_until_complete(exec_main(repo))
    except KeyboardInterrupt:
        logging.exception('KeyboardInterrupt captured.')
    except:
        logging.exception('Exception captured.')
    finally:
        logging.info('Closing the loop')
        loop.run_until_complete(close_main(repo))
        loop.close()
    logging.info('Shutting down')