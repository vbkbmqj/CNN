require 'torch'
require 'nn'
require 'optim'
require 'src.utils'

require 'src.model_1'


local arg = getArg()
arg.model = arg.model or 1
--arg.load = 'last'

Option = {
  arg=arg
  ,funOption={}
  --,optimState={}
  }



Train = {}


function Train:trainOneEpoch(funInitEpoch)
  local funOptim, optimState, funEndBatch, funEndEpoch, net, criterion, iterBatch = funInitEpoch()

  local params,gradParams = net:getParameters()

  local bEnd = false
  local function feval(x)
    if params ~= x then params:copy(x); print("++++++++++++ params changed!! +++++++++"); end -- copy x? copy y?
    
    local k, inputs,targets = iterBatch()
    if not k then bEnd = true; return 0, gradParams:zero(); end

    --net:zeroGradParameters();
    gradParams:zero()
    local outputs = net:forward(inputs)
    local err = criterion:forward(outputs, targets)
    local grad = criterion:backward(outputs, targets)
    net:backward(inputs, grad);

    funEndBatch(outputs, targets, k, err);
    return err, gradParams;
  end

  repeat
    funOptim(feval, params, optimState, optimState.state)
  until bEnd

  return funEndEpoch()
end





function Train:train(Option)
  --funInitModel, funInitEpoch, funEndModel
  local tbFunTrain = {changeOption(Option)}
  local tbFunNew = {table.unpack(tbFunTrain)}

  assert(tbFunTrain[1], 'loadOption.lua error. ')
  
  local function callFun(nFunInd)
    nFunInd = nFunInd or 2
    local ret = {pcall(tbFunNew[nFunInd])}
    if ret[1] then
      tbFunTrain[nFunInd] = tbFunNew[nFunInd]
      return select(2, table.unpack(ret))
    else
      print('call tbFunTrain[' .. nFunInd .. '] (' .. tostring(tbFunTrain[nFunInd]) .. ') error. call previous function instead.')
      return tbFunTrain[nFunInd]()
    end
  end


  callFun(1)

  while self:trainOneEpoch(callFun) do
    tbFunNew[1],tbFunNew[2],tbFunNew[3] = changeOption(Option)
  end

  callFun(3)

  print("++++++++++++++++++++++++++++ train end ++++++++++++++++++++++++++++");
end



Train:train(Option);


