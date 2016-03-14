require 'MultiCrossEntropyCriterion'
--require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nn'
data = require 'data'
dir = 'data/'
X,Y = data.storeXY(dir,28,224,'captchaImage.')
X,Y = data.loadXY(dir)
--Y = Y:cuda()
local sample_data = 500

print(Y[sample_data])

model = torch.load('trained.t7')
ct = torch.load('ct.t7')

model:evaluate()
model:cuda()

local batch = X[{{sample_data-1, sample_data}}]:cuda()
print(batch:size())
--torch.reshape(batch,10)
out = model:forward(batch)

local tmp, maxoutput = out:max(3)
--print (maxoutput)
--print(tmp)
--print (maxoutput)

maxoutput=maxoutput:double()
print()
print()
print(maxoutput[2])
print('Ratio correct: ' .. maxoutput[2]:eq(Y[sample_data]):sum()/maxoutput[2]:size(1))

local outseq = {}
for i=1, Y[sample_data]:size(1) do
   table.insert(outseq,string.char(Y[sample_data][i]))
end
print('Output: ')
print(outseq)
--result = ct:forward(out, Y[{{16,3}}]:cuda())
--print(result)
--model2 = torch.load('/home/ubuntu/model_id.t7')
--print(model2)
