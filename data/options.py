import argparse
def option():
    # Training settings
    parser = argparse.ArgumentParser(description='CIDNet')
    parser.add_argument('--batchSize', type=int, default=16, help='training batch size') 
    parser.add_argument('--cropSize', type=int, default=256, help='image crop size (patch size)')
    parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for end')
    parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to start, >0 is retrained a pre-trained pth')
    parser.add_argument('--snapshots', type=int, default=10, help='Snapshots for save checkpoints pth')
    parser.add_argument('--lr', type=float, default=0.5e-3, help='Learning Rate')
    parser.add_argument('--gpu_mode', type=bool, default=True)  
    parser.add_argument('--threads', type=int, default=16, help='number of threads for dataloader to use')

    # choose a scheduler
    parser.add_argument('--cos_restart_cyclic', type=bool, default=True)  
    parser.add_argument('--cos_restart', type=bool, default=False)

    # warmup training
    parser.add_argument('--warmup_epochs', type=int, default=3, help='warmup_epochs') 
    parser.add_argument('--start_warmup', type=bool, default=True, help='turn False without warmup') 

    # train datasets
    parser.add_argument('--data_train_sid'          , type=str, default='./datasets/SID/Train')
    parser.add_argument('--data_train_lol_blur'     , type=str, default='your data path')
    parser.add_argument('--data_train_lol_v1'       , type=str, default='your data path')
    parser.add_argument('--data_train_lolv2_real'   , type=str, default='your data path')
    parser.add_argument('--data_train_lolv2_syn'    , type=str, default='your data path')
    parser.add_argument('--data_train_SID'          , type=str, default='your data path')
    parser.add_argument('--data_train_SICE'         , type=str, default='your data path')

    # validation input
    parser.add_argument('--data_val_sid'          , type=str, default='your data path')
    parser.add_argument('--data_val_lol_blur'       , type=str, default='your data path')
    parser.add_argument('--data_val_lol_v1'         , type=str, default='your data path')
    parser.add_argument('--data_val_lolv2_real'     , type=str, default='your data path')
    parser.add_argument('--data_val_lolv2_syn'      , type=str, default='your data path')
    parser.add_argument('--data_val_SID'            , type=str, default='your data patht')
    parser.add_argument('--data_val_SICE_mix'       , type=str, default='your data path')
    parser.add_argument('--data_val_SICE_grad'      , type=str, default='your data path')

    # validation groundtruth
    parser.add_argument('--data_valgt_sid'          , type=str, default='your data path')
    parser.add_argument('--data_valgt_lol_blur'     , type=str, default='your data path')
    parser.add_argument('--data_valgt_lol_v1'       , type=str, default='your data path')
    parser.add_argument('--data_valgt_lolv2_real'   , type=str, default='your data path')
    parser.add_argument('--data_valgt_lolv2_syn'    , type=str, default='your data path')
    parser.add_argument('--data_valgt_SID'          , type=str, default='your data path')
    parser.add_argument('--data_valgt_SICE_mix'     , type=str, default='your data path')
    parser.add_argument('--data_valgt_SICE_grad'    , type=str, default='your data path')
    parser.add_argument('--val_folder', default='./results/', help='Location to save validation datasets')
    
    # loss weights
    parser.add_argument('--HVI_weight', type=float, default=0.7)
    parser.add_argument('--L1_weight', type=float, default=1.2)
    parser.add_argument('--D_weight',  type=float, default=0.5)
    parser.add_argument('--E_weight',  type=float, default=0)                                                                                                                              
    parser.add_argument('--P_weight',  type=float, default=0.03)

    # choose which dataset you want to train, please only set one "True"
    parser.add_argument('--lol_v1', type=bool, default= True)
    parser.add_argument('--lolv2_real', type=bool, default=False)
    parser.add_argument('--lolv2_syn', type=bool, default=False)

    parser.add_argument('--SID', type=bool, default=False)

    return parser
