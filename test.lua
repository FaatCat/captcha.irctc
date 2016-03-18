function output_to_label(output)
   local len = output:size(1)
   --print(output)
   local label = ""
   for i=1, len do
      local char = output[i][1]
      if char<=10 then
         label = label .. string.char(char - 1 + string.byte('0'))
      else
         label = label .. string.char(char - 10 - 1 + string.byte('A'))
      end
   end
   return label
end

require 'MultiCrossEntropyCriterion'
--require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nn'

model = torch.load(arg[2])
--ct = torch.load('ct.t7')

model:evaluate()
model:cuda()

require 'image'


local batch = image.load(arg[1]):cuda()

out = model:forward(batch)
--print(out:size())
local tmp, maxoutput = out:max(3)
maxxoutput=maxoutput:double()
--print(maxoutput)

local outseq = output_to_label(maxoutput[1])
--print('Output: ')
print(outseq)
