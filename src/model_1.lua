unpack = unpack or table.unpack;

require 'torch'
require 'nn'
require 'rnn'
require 'nngraph'

require 'src.utils'



dnnModel[1] = function(Option)
  --[[
  --]]

  -- optimState = {
  --   optName = 'adadelta',
  --   rho = 0.95,
  --   eps = 1e-9,
  
  --   state = {}
  -- }
  -- optimState.func = optim[optimState.optName]
  -- Option.optimState = optimState;


  local net = nn.Sequential();

  net:add(nn.SpatialConvolution(3,6,5,5));
  net:add(nn.ReLU());
  net:add(nn.SpatialMaxPooling(2,2,2,2));

  net:add(nn.SpatialConvolution(6,16,5,5));
  net:add(nn.ReLU())
  net:add(nn.SpatialMaxPooling(2,2,2,2))

  --net:add(nn.View(-1));
  net:add(nn.View(16*5*5))
  net:add(nn.Linear(16*5*5,120))
  net:add(nn.ReLU())
  net:add(nn.Linear(120,84))
  net:add(nn.ReLU())
  net:add(nn.Linear(84,10))

  net:add(nn.LogSoftMax());
  print(net)

  local criterion = nn.ClassNLLCriterion(nil, true)

 

  do
    local params,gradParams = net:getParameters()

    local input = torch.Tensor(3,32,32)
    input = torch.Tensor(1,3,32,32)
    input = torch.DoubleTensor(1,3,32,32)
    --input = torch.FloatTensor(1,3,32,32)
    local target = torch.Tensor({1})

    local output = net:forward(input);
    local err = criterion:forward(output, target)
    local gradInput = criterion:backward(output, target)
    net:backward(input, gradInput)
  end

  return Option, net, criterion;
end


---SoftPlus


