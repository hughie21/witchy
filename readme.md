# Witchy

This is a python library for file processing which allow you to modify files in a simple way. Also, it provide some function that allow you to hidden the information into the image, like an article or another image.

# Install
```
pip install witchy
```

# Usage
## basic function

```python
from witchy import File # load the library

file = File("path/to/your/file") # create a File object

print(file) # show the information of the file

create_time = file("ctime") # get the attribute of the file

file["ctime"] = "2020-01-01 00:00:00" # modify the attribute of the file

file.append(b'\x00\x00\x00\x00') # append data to the file

file.save("path/your/file") # save the modified file
```

## steganography function
```python
from witchy import File
from steganography import DCT, LSB

# encode
file = File('your_path_image.png')
encoder = LSB(lag=0)
wm = "hello world"
encoder.embed(file=file, wm=wm, mode='text', out_name='output.png')

# decode
wm_shape = 88 # the length of watermark's bits (8bits)
wm = wm.extract(filename='output.png', wm_shape=wm_shape, mode='text')
print(wm) # hello world
```

