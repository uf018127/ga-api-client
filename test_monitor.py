from ga_api_package.ga_api_client import Repository, Tenant
import asyncio
import logging
import pandas as pd

async def open_main():
    return Repository('https://rdlab-214.genie-analytics.com/api', 'api@default', 'api123!@#')

async def exec_main(repo):
    async def tick(ts):
        nonlocal df
        dts = pd.date_range(ts, periods=1, freq='5T')
        df_input = await pipe.read_data(dts)
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
    dts = pd.date_range(start=start, end=end, freq='5T')
    await dset.patch_data(dts)

    # monitor latest
    end = start + pd.Timedelta('24H')
    dts = pd.date_range(start=start, end=end, freq='5T')
    df = await pipe.read_data(dts, columns=columns)
    task = await dset.monitor_data(end, tick)
    await asyncio.gather(task);

async def close_main(repo):
    await repo.close()
    await asyncio.sleep(0.250)

if __name__ == '__main__':
    logging.basicConfig(
        format='▸ %(asctime)s %(levelname)s %(filename)s:%(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.WARNING
    )
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