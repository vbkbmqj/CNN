unpack = unpack or table.unpack;

require 'torch'
require 'nn'
require 'rnn'
require 'nngraph'

--require 'src.utils'

--require 'dp'
dl = require 'dataload'
local train,valid,test = dl.loadCIFAR10()

--ds = dp.Mnist{input_preprocess= {dp.Standardize()}}

--inputSize = ds:imageSize('c')
inputSize = 1
channelSize = {64, 128}
dropoutProb = {0.2, 0.5, 0.5}
hiddenSize = {}

kernelSize = {5,5,5,5}
kernelStride = {1,1,1,1}
poolSize={2,2,2,2}
poolStride={2,2,2,2}

depth = 1;
cnn = nn.Sequential()


for i=1,#channelSize do
    cnn:add(nn.PrintSize("SpatialDropout"))
    cnn:add(nn.SpatialDropout(dropoutProb[depth]));

    cnn:add(nn.PrintSize("SpatialConvolution"))
    cnn:add(nn.SpatialConvolution(inputSize, channelSize[i], kernelSize[i], kernelSize[i], kernelStride[i], kernelStride[i], math.floor(kernelSize[i]/2), math.floor(kernelSize[i]/2)))

    cnn:add(nn.PrintSize("SpatialBatchNormalization"))
    cnn:add(nn.SpatialBatchNormalization(channelSize[i]))

    cnn:add(nn.PrintSize("SpatialMaxPooling"))
    cnn:add(nn.SpatialMaxPooling(poolSize[i], poolSize[i], poolStride[i], poolStride[i]))
    inputSize = channelSize[i]
    depth = depth + 1
end


cnn:add(nn.PrintSize("layer 2"))

--outsize = cnn:outside{1, ds:imageSize('c'), ds:imageSize('h'), ds:imageSize('w')}
--outsize = torch.LongTensor{1,128,4,4}
--inputSize = outsize[2]*outsize[3]*outsize[4]
inputSize = 8192

--cnn:insert(nn.Convert(ds:ioShapes(), 'bchw'), 1)
--cnn:insert(nn.Convert("bhwc", 'bchw'), 1)

cnn:add(nn.Collapse(3))
cnn:add(nn.PrintSize("Collapse"))

for i,hids in ipairs(hiddenSize) do
    cnn:add(nn.Dropout(dropoutProb[depth]))
    cnn:add(nn.PrintSize("dp"))
      
    cnn:add(nn.Linear(inputSize, hids))
    cnn:add(nn.PrintSize("ln"))
    
    cnn:add(nn.BatchNormalization(hids))
    cnn:add(nn.PrintSize("bn"))
    
    cnn:add(nn.Tanh())
    cnn:add(nn.PrintSize("t"))
    
    intputSize = hids
    depth = depth + 1
end

cnn:add(nn.Dropout(dropoutProb[depth]))
cnn:add(nn.PrintSize("dp"))
--cnn:add(nn.Linear(inputSize, #ds:classes()))
cnn:add(nn.Linear(inputSize, 10))
cnn:add(nn.PrintSize("ln"))

cnn:add(nn.LogSoftMax())
cnn:add(nn.PrintSize("lsm"))

print(cnn)

criterion = nn.ClassNLLCriterion()

input = torch.randn(2, 1, 32, 32)
--target = torch.randomperm(10)
--target = torch.range(1,10):view(input:size(1), -1)
target = torch.LongTensor{1, 2}
print("target++++++++++++++", target)

output = cnn:forward(input)
print("cnn forward");

err = criterion:forward(output, target)
print("criterion forward");
grad = criterion:backward(output, target)
cnn:backward(input, grad)