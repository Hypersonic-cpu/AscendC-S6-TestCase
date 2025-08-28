import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
torch.npu.config.allow_internal_format = False
import numpy as np
import sys  

torch.set_printoptions(precision=9)

U = np.float16
T = torch.float32

case_data = {
    'case3': {
        'input': torch.empty(80).uniform_(-10000, 10000).to(torch.bfloat16), 
        'other': torch.empty(80).uniform_(-1000, 100000).to(torch.bfloat16)
    }, 
    'case4': {
        'input': torch.empty(80, 1, 1280, 2).uniform_(-0.05, 0.05).to(torch.bfloat16), 
        'other': torch.empty(120, 1, 2).uniform_(-0.05, 0.05).to(torch.bfloat16)
    }, 
    # 'case6': {
    #     'input':np.random.uniform(-10, 10, [3, 1]).astype(np.float32),
    #     'other':np.random.uniform(-10, 10, [3, 2]).astype(np.float32)
    # }, 
    'case2': {
        'input':np.random.uniform(-10, 10, [10, 2, 3, 4]).astype(np.float32),
        'other':np.random.uniform(-10, 10, [10, 2, 3, 4]).astype(np.float32)
    }, 
    # 'case1': {
    #     'input':np.random.uniform(-10, 10, [30]).astype(U),
    #     'other':np.random.uniform(-10, 10, [30]).astype(U)
    # }
    'case1': {
        'input':np.random.uniform(-10, 10, [2048]).astype(np.float16),
        'other':np.random.uniform(-10, 10, [2048]).astype(np.float16)
    }, 
    'case11': {
        'input':np.random.uniform(-10, 10, [2, 1, 3]).astype(np.float32),
        'other':np.random.uniform(-10, 10, [1, 4, 1]).astype(np.float32)
    }, 
    'case12': {
        'input':np.random.uniform(-10, 10, [1, 3, 1, 4]).astype(np.float32),
        'other':np.random.uniform(-10, 10, [2, 1, 5, 1]).astype(np.float32)
    }, 
    'case10': {
        'input':np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32),
        'other':np.random.uniform(-10, 10, [1, 1]).astype(np.float32)
    }, 
    'case13': {
        'input':np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32),
        'other':np.random.uniform(-10, 10, [3, 4]).astype(np.float32)
    }, 
    'case14': {
        'input':np.random.uniform(-10, 10, [1]).astype(np.float32),
        'other':np.random.uniform(-10, 10, [220, 301, 486]).astype(np.float32)
    }, 
    'case15': {
        'input':np.random.uniform(-10, 10, [1]).astype(np.float32),
        'other':np.random.uniform(-10, 10, [220, 301, 486]).astype(np.float32)
    }, 
}

def verify_result(real_result, golden):
    if golden.dtype == torch.float32:
        loss = 1e-5  # 容忍偏差，一般fp32要求绝对误差和相对误差均不超过万分之一
    else:
        loss = 1e-4  # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过千分之一
    minimum = 10e-10

    a = real_result - golden  # 计算运算结果和预期结果偏差
    rtol_diff = torch.abs(a)   # 计算运算结果和预期结果偏差绝对值
    golden = torch.where(golden == 0, minimum, golden) # 替换0值为10e-10，防止除零错误
    atol_diff = torch.abs(torch.div(a, golden))  # 计算运算结果和预期结果偏差相对误差
    error_result = (rtol_diff > loss) & (atol_diff > loss)  # 计算运算结果和预期结果偏差是否同时超出误差范围
    err_num = torch.sum(error_result == True) # 相对偏差和绝对偏差均超出预期的元素个数
    if real_result.numel() * loss < err_num:  # 误差超出预期时返回打印错误，返回对比失败
            print("[ERROR] result error")
            return False
    print("test pass")
    return True

    
class TestCustomOP(TestCase):
    def test_custom_op_case(self,num):
        print(num)
        caseNmae='case'+str(num) 
        input_x = None
        input_other = None
        if int(num) == 3 or int(num) == 4:
            input_x = case_data[caseNmae]["input"]
            input_other = case_data[caseNmae]["other"]
        else:
            input_x = torch.from_numpy(case_data[caseNmae]["input"])
            input_other = torch.from_numpy(case_data[caseNmae]["other"])
        
        output = torch.hypot(input_x, input_other)
        #
        # abs1 = torch.abs(input_x)
        # abs2 = torch.abs(input_other)
        # mi   = torch.min(abs1, abs2).to(T)
        # ma   = torch.max(abs1, abs2).to(T)
        # frac = mi / ma
        # output = (torch.min(abs1, abs2) / torch.max(abs1, abs2) + 1).to(torch.float16)
        # output_npu = (torch.sqrt(frac * frac + 1) * ma).to(torch.float16)
        output_npu = custom_ops_lib.custom_op(input_x.npu(), input_other.npu())
        print ("Input X", input_x, sep='\n')
        print ("Input O", input_other, sep='\n')
        print ("Golden ", output, sep='\n')
        print ("Custom ", output_npu.cpu(), sep='\n')
        print ("RDelta ", (output_npu.cpu() - output) / (output + 1e-10), sep='\n')
        if output_npu is None:
            print(f"{caseNmae} execution timed out!")
        else:

            if verify_result(output_npu.cpu(), output):
                print(f"{caseNmae} verify result pass!")
            else:
                print(f"{caseNmae} verify result failed!")

if __name__ == "__main__":
    TestCustomOP().test_custom_op_case(sys.argv[1])
    
