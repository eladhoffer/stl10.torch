require 'torch'
require 'image'

stl10 = torch.load('stl10-train.t7')
image.display{image=stl10.data[{{1,100},{},{},{}}], nrow=40}
