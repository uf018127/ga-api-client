# GenieAnalytics Python Client

This package is the official Python client to access GenieAnalytics API server.

## Installation

```
pip install ga-api-client-uf018127
```

## Import Modules

```python
from ga_api import Repository, HyperLogLog
```

## Top Level Pattern

```python
from ga_api import Repository, System, HyperLogLog
import asyncio
import pandas as pd

async def main():
    try:
        repo = Repository('https://rdlab-214.genie-analytics.com/api', 'api', 'default', 'api123!@#')
        # ...
        # access API server to do whatever you need
        # ...
    finally:
        await repo.close()

if __name__ == '__main__':
    asyncio.run(main())
```
