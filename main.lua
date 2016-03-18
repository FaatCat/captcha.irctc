require 'cunn'
data = require 'data'
dir = 'data/'
seq_len, num_classes = 10, 36
dataH, dataW = 120, 240
X,Y = data.storeXY(dir,dataH,dataW,'captchaImage.')
print('Loading images...')
X,Y = data.loadXY(dir)
print('Loaded ' .. X:size(1) .. ' images')

Xt,Yt,Xv,Yv = data.split(X,Y,50)
--print(Yt[1])
models = require 'models'
net,ct = models.cnnModel(seq_len, num_classes, dataH,dataW)
net=net:cuda()
ct = ct:cuda()
batchSize = 16
train = require 'train'
sgd_config = {
	   learningRate = 0.1,
	      momentum = 0.9,
}
print('STARTING TRAINING...')

for i=1,20 do
   train.sgd(net,ct,Xt,Yt,Xv,Yv,1,sgd_config,batchSize)
   
   torch.save('trained.t7', net)
   torch.save('ct.t7', ct)
end
