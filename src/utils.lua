unpack = unpack or table.unpack
require 'rnn'
require 'nngraph'
require 'torch'
require 'os'
path = require 'path'

dnnModel = dnnModel or {}


function getArg()
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('train CIFAR10')
  cmd:option('-bUseCuda','1', 'use cuda to train the model')
  cmd:option('-bLogToFile', 'false', 'weather log to terminal or file')
  cmd:option('-nMaxEpoch','500', "max epoch to train")
  cmd:option('-logHead', '', 'string to be appended ahead of each line of log')
  cmd:option('-model','1', 'create model to train')
  cmd:option('-load','','checkpoint pathname to load and train ("last" for last saved checkpoint)')
  cmd:option('-checkDir', '/home/wg/Documents/check', 'path to store log and checkpoint(use absolute path)')
  --cmd:option('-optName', 'adadelta', 'optim algorithm to be used')
  cmd:option('-nBatchSize', '512', 'optim algorithm to be used')
  cmd:option('-bSaveModel','false', 'save model parameters each batch')
  cmd:text()
  par = cmd:parse(arg)
  
  par.bUseCuda = not (par.bUseCuda == 'false' or par.bUseCuda == '0')
  par.bLogToFile = not (par.bLogToFile == 'false' or par.bLogToFile == '0')
  par.logHead = par.logHead ~= "" and par.logHead or io.popen('hostname -s'):read()
  par.model = tonumber(par.model)
  par.nCurrentEpoch = 0
  par.nMaxEpoch = tonumber(par.nMaxEpoch)
  par.nBatchSize = tonumber(par.nBatchSize)
  par.bSaveModel = not (par.bSaveModel == 'false' or par.bSaveModel == '0')
  return par
end


function loadOption(opt)
  local p = path.join(opt.arg.checkDir, tostring(opt.arg.model))
  local destf = path.join(p, "changeOption.lua")
  local srcf = './src/changeOption.lua'
  
  if not path.exists(destf) then
    print("warning: " .. destf .. " not found, duplicating from " .. srcf .. ".")
    os.execute('mkdir -p ' .. p)
    local b, msg, n = os.execute('cp ' .. srcf .. ' ' .. p)
    assert(b, msg .. ": " .. n)
  end

  --local ret = {pcall(dofile, srcf)}
  local ret = {pcall(dofile, './src/changeOption.lua')} 
  --local ret = {pcall(dofile, destf)} 

  if ret[1] then
    return select(2,table.unpack(ret))
  end
  print("warning: " .. destf .. " error: " .. (ret[2] or ""))
end


function  changeOption(opt)
  local rom = {loadOption(opt)}
  if not rom[1] then return nil; end

  local ret = {pcall(rom[1], opt)}
  if ret[1] then
    return select(2,table.unpack(ret))
  end
  local destf = path.join(opt.arg.checkDir, tostring(opt.arg.model), "changeOption.lua")
  print("warning: " .. destf .. " error: " .. (ret[2] or ""))
end



function loadModel(opt)
  if opt.arg.load == '' then
    print('----------------loading model: ' .. opt.arg.model)
    return dnnModel[opt.arg.model](opt)
  end

  local file = path.join(opt.arg.checkDir, tostring(opt.arg.model), opt.arg.load)
  print(string.format("------------------loading file: %s\n%s", opt.arg.load, io.popen('ls -la ' .. file):read()))
  return table.unpack(torch.load(file))
end



function saveModel(err, checkPoint)
  local opt = checkPoint[1];

  local dir = path.join(opt.arg.checkDir, tostring(opt.arg.model))
  local filename = string.format('model%.3f_epoch%03d_%s_err%.6f.t7', opt.arg.model, opt.arg.nCurrentEpoch + 1, os.date('%y-%m-%d_%X',os.time()), err)
  local pathname = path.join(dir, filename)
	
  os.execute("mkdir -p " .. dir)
  torch.save(pathname , checkPoint);
  os.execute(string.format('ln -sf %s %s/last', filename, dir))

  print('saved checkpoint:', pathname)
end





function clip(params,delta)
  delta = delta or 3
  
  if torch.isTensor(params) then
	  local mean, stdv = params:mean(),params:std()
--	  print(mean, stdv)
	  local low,high = mean-stdv, mean+stdv
--     -------------------------------------------------------------------
-- --  local cnt = 0
--     params:apply(function(v)
-- --      if v< low or v > high then
-- --        cnt = cnt + 1
-- --      end
--       return v<low and low or v>high and high or v;
--     end)
-- --    print(cnt / params:nElement())
--     -------------------------------------------------------------------
    nn.HardTanh(low,high,true)(params)
--     -------------------------------------------------------------------

  elseif type(params) == 'table' then
    for k,v in pairs(params) do
      clip(v, delta)
    end
  else
    error('unknown type ------------------------------------------------------ src.utils.clip')
  end
end
 