import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
from tqdm import tqdm
from data.data import *
from torchvision import transforms
from torch.utils.data import DataLoader
from loss.losses import *
from model.MCBCG_Net import MCBCG_Net

eval_parser = argparse.ArgumentParser(description='Eval')
eval_parser.add_argument('--best', action='store_true', help='trained with perceptual loss')
eval_parser.add_argument('--lol', action='store_true', help='output lolv1 dataset')
eval_parser.add_argument('--lol_v2_real', action='store_true', help='output lol_v2_real dataset')
eval_parser.add_argument('--lol_v2_syn', action='store_true', help='output lol_v2_syn dataset')

eval_parser.add_argument('--unpaired', action='store_true', help='output unpaired dataset')
eval_parser.add_argument('--DICM', action='store_true', help='output DICM dataset')
eval_parser.add_argument('--LIME', action='store_true', help='output LIME dataset')
eval_parser.add_argument('--MEF', action='store_true', help='output MEF dataset')
eval_parser.add_argument('--NPE', action='store_true', help='output NPE dataset')
eval_parser.add_argument('--VV', action='store_true', help='output VV dataset')
eval_parser.add_argument('--alpha', type=float, default=1.0)
eval_parser.add_argument('--unpaired_weights', type=str, default='')

ep = eval_parser.parse_args()


def eval(model, testing_data_loader, model_path, output_folder,norm_size=True,LOL=False,v2=False,unpaired=False,alpha=1.0):
    torch.set_grad_enabled(False)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    print('Pre-trained model is loaded.')
    model.eval()
    print('Evaluation:')
    if LOL:
        model.trans.gated = True
    elif v2:
        model.trans.gated2 = True
        model.trans.alpha = alpha
    elif unpaired:
        model.trans.alpha = alpha
    for batch in tqdm(testing_data_loader):
        with torch.no_grad():
            if norm_size:
                input, name = batch[0], batch[1]
            else:
                input, name, h, w = batch[0], batch[1], batch[2], batch[3]
            
            input = input.cuda()
            output = model(input) 
            
        if not os.path.exists(output_folder):          
            os.mkdir(output_folder)  
            
        output = torch.clamp(output.cuda(),0,1).cuda()
        if not norm_size:
            output = output[:, :, :h, :w]
        
        output_img = transforms.ToPILImage()(output.squeeze(0))
        output_img.save(output_folder + name[0])
        torch.cuda.empty_cache()
    print('===> End evaluation')
    if LOL:
        model.trans.gated = False
    elif v2:
        model.trans.gated2 = False
    torch.set_grad_enabled(True)
    
if __name__ == '__main__':
    
    cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, or need to change CUDA_VISIBLE_DEVICES number")
    
    if not os.path.exists('./output'):          
            os.mkdir('./output')  
    
    norm_size = True
    num_workers = 1
    alpha = None
    if ep.lol:
        eval_data = DataLoader(dataset=get_eval_set(""), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = ''
        if ep.best:
            weight_path = ''
        else:
            weight_path = ''
        
            
    elif ep.lol_v2_real:
        eval_data = DataLoader(dataset=get_eval_set(""), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = ''
        if ep.best_GT_mean:
            weight_path = ''
            alpha = 0.84
        elif ep.best_PSNR:
            weight_path = ''
            alpha = 0.8
        elif ep.best_SSIM:
            weight_path = ''
            alpha = 0.82
            
    elif ep.lol_v2_syn:
        eval_data = DataLoader(dataset=get_eval_set(""), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = ''
        if ep.best:
            weight_path = ''
        else:
            weight_path = ''
            
    elif ep.SICE_grad:
        eval_data = DataLoader(dataset=get_SICE_eval_set(""), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = ''
        weight_path = ''
        norm_size = False
        
    elif ep.SICE_mix:
        eval_data = DataLoader(dataset=get_SICE_eval_set(""), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = ''
        weight_path = ''
        norm_size = False
    
    elif ep.unpaired: 
        if ep.DICM:
            eval_data = DataLoader(dataset=get_SICE_eval_set(""), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = ''
        elif ep.LIME:
            eval_data = DataLoader(dataset=get_SICE_eval_set(""), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = ''
        elif ep.MEF:
            eval_data = DataLoader(dataset=get_SICE_eval_set(""), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = ''
        elif ep.NPE:
            eval_data = DataLoader(dataset=get_SICE_eval_set(""), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = ''
        elif ep.VV:
            eval_data = DataLoader(dataset=get_SICE_eval_set(""), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = ''
        alpha = ep.alpha
        norm_size = False
        weight_path = ep.unpaired_weights
        
    eval_net = MCBCG_Net().cuda()
    eval(eval_net, eval_data, weight_path, output_folder,norm_size=norm_size,LOL=ep.lol,v2=ep.lol_v2_real,unpaired=ep.unpaired,alpha=alpha)

