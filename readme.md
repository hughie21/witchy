# Witchy

This is a python library for file processing which allow you to modify files in a simple way. Also, it provide some function that allow you to convert the format of the file or doing compression on it.

# Install
```
pip install witchy
```
> **Attention: make sure you have installed the `FFmpeg` and correctly set the path of your local environment**.

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

## convert function
```python
from witchy import File

file = File("yourpic.jpg")

file.convert("path/output.png","PNG") # convert jpg to png format
```

