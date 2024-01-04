# Copyright (c) 2022 Oracle and/or its affiliates.

'''
@author jasperan
This file uses web sockets to communicate Live Client API data from one endpoint to the other. You can virtually put your consumer 
wherever you want, and let it process the incoming data from the producer.
''' 
import asyncio
from websockets import connect
import sys, os
import argparse
from rich import print


cli_parser = argparse.ArgumentParser()
cli_parser.add_argument('-i', '--ip', type=str, help='IP address to make requests to', default='127.0.0.1', required=False)
args = cli_parser.parse_args()

async def get_info(uri):
    async with connect(uri) as websocket:
        while True:
            await websocket.send("get_img")
            img_b64 = await websocket.recv()

            print(img_b64)


def main():
    asyncio.run(get_info('ws://{}:8001'.format(args.ip)))



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
