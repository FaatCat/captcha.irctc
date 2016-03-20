require 'cunn'
data = require 'data'
dir = 'data/'
seq_len, num_classes = 10, 36
dataH, dataW = 120, 240
training_set = 0.85

--print('creating data.t7..')
--X,Y = data.storeXY(dir,dataH,dataW,'captchaImage.')

print('Loading images...')
X,Y = data.loadXY(dir)
print('Loaded ' .. X:size(1) .. ' images')

Xt,Yt,Xv,Yv = data.split(X,Y,math.floor(X:size(1)*0.15))


--[[-- Split training data
Y = data.loadY('data/')
local dataNum = Y:size(1)
print(dataNum .. ' data found.')

local trainStart = 1
local numTrain = math.floor(dataNum*training_set)
local trainEnd = trainStart+numTrain-1
local valStart = trainEnd+1
local valEnd = dataNum

print('TrainingData: ' .. trainStart .. ' to ' .. trainEnd)
print('ValidationData: ' .. valStart .. ' to ' .. valEnd)

Xt = data.loadXn(dir, 'captchaImage.', trainStart, trainEnd, dataH, dataW)
Yt = Y.index(1,trainStart)
Xv = data.loadXn(dir, 'captchaImage.', valStart, valEnd)
Yv = Y.index(valStart, valEnd)
]]--
--print(Yt[1])
models = require 'models'
net,ct = models.cnnModel(seq_len, num_classes, dataH,dataW)
net=net:cuda()
ct = ct:cuda()
batchSize = 1
train = require 'train'
sgd_config = {
	   learningRate = 0.1,
	      momentum = 0.9,
}


-- simulate a fixed memory on the GPU
local freemem = cutorch.getMemoryUsage()
local neededmem = 0.8 * 1024 * 1024 * 1024
if freemem - neededmem > 4 then
   reserved_tensor = torch.CudaTensor((freemem - neededmem) / 4):fill(0)
end
print('Free memory: ' .. cutorch.getMemoryUsage())


print('STARTING TRAINING...')

for current_epoch_for_display=1,20 do
   train.sgd(net,ct,Xt,Yt,Xv,Yv,current_epoch_for_display,sgd_config,batchSize,dataH, dataW)
   
   torch.save('trained.t7', net)
   torch.save('ct.t7', ct)
end
