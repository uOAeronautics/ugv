"""Bounded grayscale camera cache and PNG preview, using only the standard library."""
import struct
import threading
import time
import zlib


def gray_png(width,height,data,step=None):
    step=step or width
    if width<=0 or height<=0 or step<width or len(data)<step*height:
        raise ValueError('Invalid grayscale camera frame')
    def chunk(name,body):
        return struct.pack('!I',len(body))+name+body+struct.pack('!I',zlib.crc32(name+body)&0xffffffff)
    pixels=b''.join(b'\0'+data[y*step:y*step+width] for y in range(height))
    return (b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',width,height,8,0,0,0,0))+
            chunk(b'IDAT',zlib.compress(pixels,1))+chunk(b'IEND',b''))


class CameraStream:
    def __init__(self):
        self.lock=threading.Lock();self.frame=None;self.sequence=0;self.received=0
        self.cached_sequence=-1;self.cached=None

    def update(self,message):
        # Native sensor is L8. Never silently reinterpret color/depth bytes.
        if message.pixel_format_type!=1 or message.width>4096 or message.height>4096:return
        with self.lock:
            self.sequence+=1;self.received=time.monotonic()
            self.frame=(message.width,message.height,bytes(message.data),message.step,
                        message.header.stamp.sec+message.header.stamp.nsec*1e-9)

    def status(self):
        with self.lock:
            return dict(streaming=self.frame is not None and time.monotonic()-self.received<2,
                sequence=self.sequence,width=self.frame[0] if self.frame else 0,height=self.frame[1] if self.frame else 0,
                t_sim_s=self.frame[4] if self.frame else None)

    def png(self):
        with self.lock:
            sequence=self.sequence;frame=self.frame
            if frame is None:return None
            if sequence==self.cached_sequence:return self.cached
        image=gray_png(*frame[:4])
        with self.lock:self.cached_sequence=sequence;self.cached=image
        return image
