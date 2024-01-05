import asyncio
from websockets.server import serve
import websockets 

b64_img = "Test"
async def echo(websocket):
    async for message in websocket:
        global b64_img
        await websocket.send(b64_img)



async def main():
    async with websockets.serve(echo, "127.0.0.1", 8765):
        print('Connected...')
        await asyncio.Future()  # run forever
    #event.set() # stop capture thread


asyncio.run(main())