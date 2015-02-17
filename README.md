Downloads STL10 dataset directly from http://cs.stanford.edu/~acoates/stl10/ and converts them to Torch tables.
Based on https://github.com/soumith/cifar.torch.git

STL-10 format
---------------
Writes 3 files: stl10-train.t7, stl10-test.t7, stl10-unlabeled.t7
Each of them is a table of the form:
```lua
th> stl10 = torch.load('stl10-train.t7')
th> print(stl10)
{
        data : ByteTensor - size: 50000x3x96x96
        label : ByteTensor - size: 50000
}
```
The Unlabeled file has only the data field
