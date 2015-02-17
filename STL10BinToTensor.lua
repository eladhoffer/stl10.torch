require 'torch'

os.execute('wget -c http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz')
os.execute('tar -xvf stl10_binary.tar.gz')
local function convertSTL10BinToTorchTensor(inputFname, inputLabelsFname, outputFname)
    local nSamples = 0
    local m=torch.DiskFile(inputFname, 'r'):binary()
    m:seekEnd()
    local length = m:position() - 1
    local nSamplesF = length / (3*96*96) 
    assert(nSamplesF == math.floor(nSamplesF), 'expecting numSamples to be an exact integer')
    nSamples = nSamples + nSamplesF
    m:close()

    local data = torch.ByteTensor(nSamples, 3, 96, 96)
    local index = 1
    local m=torch.DiskFile(inputFname, 'r'):binary()
    m:seekEnd()
    local length = m:position() - 1
    local nSamplesF = length / (3*96*96)
    m:seek(1)
    for j=1,nSamplesF do
        local store = m:readByte(3*96*96)
        data[index]:copy(torch.ByteTensor(store))
        index = index + 1
    end
    m:close()


    local out = {}
    
    out.data = data:transpose(3,4)

    if inputLabelsFname then
        local m=torch.DiskFile(inputLabelsFname, 'r'):binary()
        out.label = torch.ByteTensor(m:readByte(nSamplesF))
        m:close()
    end

    print(out)
    torch.save(outputFname, out)
end

convertSTL10BinToTorchTensor('stl10_binary/unlabeled_X.bin',nil,'stl10-unlabeled.t7')

convertSTL10BinToTorchTensor('stl10_binary/train_X.bin','stl10_binary/train_y.bin', 'stl10-train.t7')
convertSTL10BinToTorchTensor('stl10_binary/test_X.bin','stl10_binary/test_y.bin', 'stl10-test.t7')
