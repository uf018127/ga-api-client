from ga_api import Repository, HyperLogLog
import asyncio
import pandas as pd
import logging

async def main():
    # open repository
    repo = Repository('https://rdlab-214.genie-analytics.com/api', 'api', 'default', 'api123!@#')

    try:

        dset_id = 13579
        dset_conf = {
            'name':'example2',
            'series':'hour',
            'freq':300,
            'retainDepth':86400*7,
            'retainSize':10000000,
            'run':False
        }
        # delete dataset
        await repo.del_dataset(dset_id, missing_ok=True)
        # get dataset (create)
        dset = await repo.get_dataset(dset_id, dset_conf)
        # get dataset (read)
        dset = await repo.get_dataset(dset_id)
        # delete dataset
        await repo.del_dataset(dset_id, missing_ok=True)
        # set dataset (create)
        dset = await repo.set_dataset(dset_id, dset_conf)
        # set dataset (update)
        dset = await repo.set_dataset(dset_id, dset_conf)

        pipe_id = 1001
        pipe_conf = {
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
        }
        # delete pipeline
        await dset.del_pipeline(pipe_id, missing_ok=True)
        # get pipeline (create)
        pipe = await dset.get_pipeline(pipe_id, pipe_conf)
        # get pipeline (read)
        pipe = await dset.get_pipeline(pipe_id)
        # delete pipeline
        await dset.del_pipeline(pipe_id, missing_ok=True)
        # set pipeline (create)
        pipe = await dset.set_pipeline(pipe_id, pipe_conf)
        # set pipeline (update)
        pipe = await dset.set_pipeline(pipe_id, pipe_conf)

        # patch dataset data
        dts = pd.date_range('2022-03-11 12:00:00', end='2022-03-11 13:00:00', freq='5T')
        print(await dset.patch_data(dts))
        # poll dataset data
        print(await dset.poll_data(dts))
        # read pipeline data range
        df = await pipe.read_data(dts, columns=['ts','forward','opposite'])
        print(df)
        # read pipeline data point
        df = await pipe.read_data('2022-03-11 12:00:00', columns=['forward','opposite'])
        print(df)

        # delete pipeline
        await dset.del_pipeline(1001)
        # delete all pipelines
        await dset.del_all_pipelines()    
        # delete dataset
        await repo.del_dataset(dset_id)    

        # adhoc
        adhoc = await repo.set_adhoc(300, {
            'scope': {
                'i-field': '@controller.home',
                'i-entry': [
                    [True,'t:0:0:1',0]
                ],
                'mode': '%directional'
            },
            'metric': [
                ['$distinct', {
                    'fields':['@flow.addr.dst']
                }]
            ]
        })
        # read adhoc data range
        dts = pd.date_range('2022-03-11 12:00:00', end='2022-03-11 13:00:00', freq='5T')
        df = await adhoc.read_data(dts, 'hour', columns=['ts','forward','opposite'])
        print(df)
        # read adhoc data point
        ts = pd.Timestamp('2022-03-11 12:00:00')
        df = await adhoc.read_data(ts, 'hour', columns=['forward','opposite'])
        print(df)

    finally:
        await repo.close()

if __name__ == '__main__':
    logging.basicConfig(
        format='▸ %(asctime)s %(levelname)s %(filename)s:%(funcName)s(%(lineno)d) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        # level=logging.INFO
    )
    asyncio.run(main())