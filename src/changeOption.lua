require 'gnuplot'
dl = require 'dataload'



--g_bModelInited  = g_bModelInited or false

--g_net = g_net or "not initialized"
--g_criterion = g_criterion or "not initialized"
--g_data = g_data or "not initialized"

--g_funLastLine = g_funLastLine or 'no data'


local function changeOptionFunc(Option)
    --Option.funOption.g_bModelInited = Option.funOption.g_bModelInited or false
    Option.funOption.state = Option.funOption.state or {}

    Option.funOption.state.g_net = Option.funOption.state.g_net or "not initialized"
    Option.funOption.state.g_criterion = Option.funOption.state.g_criterion or "not initialized"
    Option.funOption.state.g_data = Option.funOption.state.g_data or "not initialized"
    Option.funOption.state.g_funLastLine = Option.funOption.state.g_funLastLine or "not initialized"

    function Option.funOption:initModel()
        local logPath = path.join(Option.arg.checkDir, tostring(Option.arg.model))

        if Option.arg.bLogToFile then
            io.output(io.open(path.join(logPath, Option.arg.logHead .. ".txt"),"a+"))  ----------------- output 
        end

        local fl = path.join(logPath, Option.arg.logHead .. "Result.txt")
        local fileResult = io.open(fl,"a+")
        self.state.g_funLastLine = function(err)
            fileResult:write(err, ',')
            fileResult:flush()
            local data = loadstring('return {' .. io.popen('tail -1 ' .. fl):read() .. '}')()
            gnuplot.plot(torch.Tensor(data)); 
        end
       
        local loadOpt, net, criterion = loadModel(Option)
        if loadOpt ~= Option then
            Option.optimState = loadOpt.optimState
            --Option.arg = loadOpt.arg
            --Option.funOption = loadOpt.funOption
        else
            fileResult:write('\n')
        end
        

        local train,valid,test = dl.loadCIFAR10()
        --train,valid,test = dl.loadMNIST()
        local data = train

        if Option.arg.bUseCuda then
            require 'cutorch'; require 'cunn';
            data.inputs = data.inputs:cuda();
            data.targets = data.targets:cuda();
            net = net:cuda()
            criterion = criterion:cuda()
        else
            net = net:float()
            criterion = criterion:float()
        end


        self.state.g_net = net
        self.state.g_criterion = criterion
        self.state.g_data = data;
    end
    

    function Option.funOption:initEpoch()
        local arg = Option.arg
        local logHead, model = arg.logHead, arg.model
        local nDataSize = self.state.g_data:size()
        local nSumCorrect = 0


        --arg.nMaxEpoch = 200
        arg.nBatchSize = 2048


        optimState = {
            optName = 'adadelta',
            rho = 0.90,
            eps = 1e-6,
            weightDecay = 0,

            state = {}
        }
        optimState = {
            optName = 'sgd'
            , learningRate = 0.001
        }

        optimState.func = optim[optimState.optName]

        Option.optimState = Option.optimState or optimState
        --Option.optimState = optimState 

        Option.optimState.learningRate = 0.1



        print('arg:', arg)
        print('optimState:', Option.optimState)


        self.state.g_data:shuffle() -- torch.randomperm(size)
        local iterBatch = self.state.g_data:subiter(arg.nBatchSize, nDataSize) --------------------------------
    
        function funEndBatch(outputs, targets, k, err)
            local nCorrect = torch.eq((({torch.max(outputs, 2)})[2]):view(-1):typeAs(targets),targets):sum()
            nSumCorrect = nSumCorrect + nCorrect
            print(string.format("%s  model%.2f  epoch%.3f %d %s correct%.5f", logHead, model, arg.nCurrentEpoch+k/nDataSize, k, os.date("%x_%X"), nCorrect/outputs:size(1)))
        end

        function funEndEpoch()
            print("model" .. arg.model, "Epoch" .. arg.nCurrentEpoch + 1, os.date("%x %X"), nSumCorrect / nDataSize);

            if Option.arg.bSaveModel then
                local state = self.state
                local net, criterion = state.g_net, state.g_criterion
                self.state = nil
                saveModel(nSumCorrect / nDataSize, {Option, net, criterion});
                self.state = state
            end


            self.state.g_funLastLine(nSumCorrect / nDataSize)

            arg.nCurrentEpoch = arg.nCurrentEpoch + 1
            return arg.nCurrentEpoch < arg.nMaxEpoch
        end   

        return Option.optimState.func, Option.optimState, funEndBatch, funEndEpoch, self.state.g_net, self.state.g_criterion, iterBatch
    end
    
    

    function Option.funOption:endModel()
        --os.execute('sudo shutdown')
    end
    
    return true;
end


local function funRet(Option)
    changeOptionFunc(Option)

    return function()
        return Option.funOption:initModel()
    end

    , function()
        return Option.funOption:initEpoch()
    end

    , function()
        return Option.funOption:endModel()
    end
end


return funRet