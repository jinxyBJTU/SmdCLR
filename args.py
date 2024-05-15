import argparse


def get_args():
    parser = argparse.ArgumentParser('Train_MoCo')

    # General args
    parser.add_argument("--seed", type=int, default= 1)

    # parser.add_argument('--temp',
    #                     type=float,
    #                     default=0.2,
    #                     help='temperture')
    
    parser.add_argument('--cuda', type=str, default='0')

    parser.add_argument('--draw_negative', type=int, default=0) 

    parser.add_argument('--semi_rate', type=float, default=0) 

    parser.add_argument('--weight_decay',
                        type=float,
                        default=1e-3,
                        help='SDG parameter.')
    
    parser.add_argument('--sgd_momentum',
                        type=float,
                        default=0.9,
                        help='SDG parameter.')

    parser.add_argument('--k',
                        type=int,
                        default=2000,
                        help='state_dict parameter.')

    parser.add_argument('--K',
                        type=int,
                        default=32768,
                        help='MoCo parameter.')

    parser.add_argument('--m',
                        type=float,
                        default=0.999,
                        help='MoCo parameter.')

    parser.add_argument('--window_size',
                        type=int,
                        default=1,
                        help='model parameter.')

    parser.add_argument('--T',
                        type=float,
                        default=0.07,
                        help='model parameter.')

    parser.add_argument('--feature_dim',
                        type=int,
                        default=128,
                        help='model parameter.')

    parser.add_argument('--training_mode',
                        type=str,
                        default="self_supervised",
                        help='training_mode.')

    parser.add_argument('--data_path',
                        type=str,
                        default="/data/Liulei/TS_TCC_main/data/FD_A",
                        help='sleepEDF_JZY,ISRUC_F3_A2,Epilepsy,Waveform_data,HAR,FD_A')

    parser.add_argument('--fs',
                        type=int,
                        default=100,
                        help='model parameter.')

    parser.add_argument('--model_name',
                        type=str,
                        default='ts-tcc',
                        help='model selected:moco,simclr,ts-tcc')

    # Training/test args
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='Training batch size.')

    parser.add_argument('--lr',
                        type=float,
                        default=1e-3, # 1e-3,
                        help='Initial learning rate.')

    parser.add_argument('--train_epochs',
                        type=int,
                        default=40,
                        help='Number of epochs for training.')
    
    parser.add_argument('--fine_epochs',
                        type=int,
                        default=40,
                        help='Number of epochs for training.')

    args = parser.parse_args()

    # must provide load_model_path if testing only

    return args
