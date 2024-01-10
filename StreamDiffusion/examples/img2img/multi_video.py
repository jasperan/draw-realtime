import threading
import video_processer

def run_main():
    import os
    #dir = '/home/ubuntu/videos/'
    dir = '../vid2vid/video/'
    for file in os.listdir(dir):
        print(file)
        video_processer.main(pathy=dir, file=file)
        #thread = threading.Thread(target=video_processer.main, args=(pathy=dir, file=file)))
        #thread.start()

run_main()