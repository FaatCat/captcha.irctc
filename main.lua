require 'cunn'
data = require 'data'
dir = 'data/'
seq_len, num_classes = 10, 36
dataH, dataW = 120, 240

print('creating data.t7..')
--X,Y = data.storeXY(dir,dataH,dataW,'captchaImage.')

print('Loading images...')
X,Y = data.loadXY(dir)
print('Loaded ' .. X:size(1) .. ' images')

Xt,Yt,Xv,Yv = data.split(X,Y,math.floor(X:size(1)*0.15))
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

for i=1,20 do
   train.sgd(net,ct,Xt,Yt,Xv,Yv,1,sgd_config,batchSize)
   
   torch.save('trained.t7', net)
   torch.save('ct.t7', ct)
end
