require 'cunn'
data = require 'data'
dir = 'data/'
X,Y = data.storeXY(dir,28,224,'captchaImage.')
X,Y = data.loadXY(dir)
print('Loaded ' .. X:size(1) .. ' images')

Xt,Yt,Xv,Yv = data.split(X,Y,50)
print(Yt[1])
models = require 'models'
net,ct = models.cnnModel()
net=net:cuda()
ct = ct:cuda()
batchSize = 16
train = require 'train'
sgd_config = {
	   learningRate = 0.1,
	      momentum = 0.9,
}
print('STARTING TRAINING...')
train.sgd(net,ct,Xt,Yt,Xv,Yv,20,sgd_config,batchSize)

torch.save('trained.t7', net)
torch.save('ct.t7', ct)
