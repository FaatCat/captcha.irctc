require 'MultiCrossEntropyCriterion'
--require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nn'
data = require 'data'
dir = 'data/'
X,Y = data.storeXY(dir,50,170,'captchaImage.')
X,Y = data.loadXY(dir)
print(Y[79])

model = torch.load('trained.t7')
ct = torch.load('ct.t7')
print('MODEL:')
for aw=1, #model.modules do
    --print(aw)
    --print(model.modules[aw])
    --print(model.modules[aw].updateOutput)
end

model:cuda()
print(model.evaluate)
print(type(X))
out = model:forward(X[{{78,79}}]:cuda())
print(out)
--result = ct:forward(out, Y[{{16,3}}]:cuda())
--print(result)
--model2 = torch.load('/home/ubuntu/model_id.t7')
--print(model2)
