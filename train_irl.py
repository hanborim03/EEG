import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import torch
import torch.optim as optim
import math
import utils
from data.data_utils import *
from data.dataloader_detection import load_dataset_detection
from data.dataloader_classification import load_dataset_classification
from data.dataloader_densecnn_classification import load_dataset_densecnn_classification
from constants import *
from args import get_args
from collections import OrderedDict
from json import dumps
from model.model import DCRNNModel_classification, DCRNNModel_nextTimePred
from model.densecnn import DenseCNN
from model.lstm import LSTMModel
from model.cnnlstm import CNN_LSTM
from tensorboardX import SummaryWriter
from tqdm import tqdm
from dotted_dict import DottedDict
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy


def main(args):

    # Get device
    args.cuda = torch.cuda.is_available()
    device = "cuda" if args.cuda else "cpu"
    print('!', device, '!')

    # Set random seed
    utils.seed_torch(seed=args.rand_seed)

    # Get save directories
    args.save_dir = utils.get_save_dir(
        args.save_dir, training=True if args.do_train else False)
    # Save args
    args_file = os.path.join(args.save_dir, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    # Set up logger
    log = utils.get_logger(args.save_dir, 'train')
    tbx = SummaryWriter(args.save_dir)
    log.info('Args: {}'.format(dumps(vars(args), indent=4, sort_keys=True)))

    # Build dataset
    log.info('Building dataset...')
    if args.task == 'detection':
        dataloaders, _, scaler = load_dataset_detection(
            input_dir=args.input_dir,
            raw_data_dir=args.raw_data_dir,
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            time_step_size=args.time_step_size,
            max_seq_len=args.max_seq_len,
            standardize=True,
            num_workers=args.num_workers,
            augmentation=args.data_augment,
            adj_mat_dir='./data/electrode_graph/adj_mx_3d.pkl',
            graph_type=args.graph_type,
            top_k=args.top_k,
            filter_type=args.filter_type,
            use_fft=args.use_fft,
            sampling_ratio=1,
            seed=123,
            preproc_dir=args.preproc_dir)
    elif args.task == 'classification':
        if args.model_name != 'densecnn':
            dataloaders, _, scaler = load_dataset_classification(
                input_dir=args.input_dir,
                raw_data_dir=args.raw_data_dir,
                train_batch_size=args.train_batch_size,
                test_batch_size=args.test_batch_size,
                time_step_size=args.time_step_size,
                max_seq_len=args.max_seq_len,
                standardize=True,
                num_workers=args.num_workers,
                padding_val=0.,
                augmentation=args.data_augment,
                adj_mat_dir='./data/electrode_graph/adj_mx_3d.pkl',
                graph_type=args.graph_type,
                top_k=args.top_k,
                filter_type=args.filter_type,
                use_fft=args.use_fft,
                preproc_dir=args.preproc_dir)
        else:
            print("Using densecnn dataloader!")
            dataloaders, _, scaler = load_dataset_densecnn_classification(
                input_dir=args.input_dir,
                raw_data_dir=args.raw_data_dir,
                train_batch_size=args.train_batch_size,
                test_batch_size=args.test_batch_size,
                max_seq_len=args.max_seq_len,
                standardize=True,
                num_workers=args.num_workers,
                padding_val=0.,
                augmentation=args.data_augment,
                use_fft=args.use_fft,
                preproc_dir=args.preproc_dir
            )
    else:
        raise NotImplementedError

    # Build model_irl
    log.info('Building model_irl...')
    if args.model_name == "dcrnn":
        model_irl = DCRNNModel_classification(
            args=args, num_classes=args.num_classes, device=device)
    elif args.model_name == "densecnn":
        with open("./model/dense_inception/params.json", "r") as f:
            params = json.load(f)
        params = DottedDict(params)
        data_shape = (args.max_seq_len*100, args.num_nodes) if args.use_fft else (args.max_seq_len*200, args.num_nodes)
        model_irl = DenseCNN(params, data_shape=data_shape, num_classes=args.num_classes)
    elif args.model_name == "lstm":
        model_irl = LSTMModel(args, args.num_classes, device)
    elif args.model_name == "cnnlstm":
        model_irl = CNN_LSTM(args.num_classes)
    else:
        raise NotImplementedError

    if args.do_train:
        if not args.fine_tune:
            if args.load_model_path is not None:
                model_irl = utils.load_model_checkpoint(
                    args.load_model_path, model_irl)
        else:  # fine-tune from pretrained model
            if args.load_model_path is not None:
                args_pretrained = copy.deepcopy(args)
                setattr(
                    args_pretrained,
                    'num_rnn_layers',
                    args.pretrained_num_rnn_layers)
                pretrained_model_irl = DCRNNModel_nextTimePred(
                    args=args_pretrained, device=device)  # placeholder
                pretrained_model_irl = utils.load_model_checkpoint(
                    args.load_model_path, pretrained_model_irl)
                model_irl = utils.build_finetune_model(
                    model_new=model_irl,
                    model_pretrained=pretrained_model_irl,
                    num_rnn_layers=args.num_rnn_layers)
            else:
                raise ValueError(
                    'For fine-tuning, provide pretrained model in load_model_path!')

        num_params_irl = utils.count_parameters(model_irl)
        log.info('Total number of trainable parameters (IRL): {}'.format(num_params_irl))

        model_irl = model_irl.to(device)
        # Train
        train(model_irl, dataloaders, args, device, args.save_dir, log, tbx)

        # Load best model after training finished
        best_path = os.path.join(args.save_dir, 'best.pth.tar')
        model_irl = utils.load_model_checkpoint(best_path, model_irl)
        model_irl=model_irl.to(device)

    # Evaluate on dev and test set
    log.info('Training DONE. Evaluating model...')
    dev_results = evaluate(model_irl,
                           dataloaders['dev'],
                           args,
                           args.save_dir,
                           device,
                           is_test=True,
                           nll_meter=None,
                           eval_set='dev')
                           

    dev_results_str = ', '.join('{}: {:.3f}'.format(k, v)
                                for k, v in dev_results.items())
    log.info('DEV set prediction results: {}'.format(dev_results_str))

    test_results = evaluate(model_irl,
                            dataloaders['test'],
                            args,
                            args.save_dir,
                            device,
                            is_test=True,
                            nll_meter=None,
                            eval_set='test',
                            best_thresh=dev_results['best_thresh'])
    
    print("test는 왜 안되는가아아앙", dev_results['best_thresh'])

    # Log to console
    test_results_str = ', '.join('{}: {:.3f}'.format(k, v)
                                 for k, v in test_results.items())
                            
    log.info('TEST set prediction results: {}'.format(test_results_str))

class IRLLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, K=1, layers_to_penalize=None):
        super(IRLLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.K = K
        self.layers_to_penalize = layers_to_penalize if layers_to_penalize is not None else []
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, model, inputs, targets, seq_lengths, supports, noise_fn):
        # 원본 입력에 대한 손실
        original_outputs = model(inputs, seq_lengths, supports).squeeze(-1)
        original_loss = self.bce_loss(original_outputs, targets)

        # 노이즈가 추가된 입력에 대한 손실
        noise_loss = 0
        dist_loss = 0

        for _ in range(self.K):
            noisy_inputs = noise_fn(inputs)
            noisy_outputs = model(noisy_inputs, seq_lengths, supports).squeeze(-1)
            noise_loss += self.bce_loss(noisy_outputs, targets)

            # 활성화 거리 계산
            original_features = model.get_features(inputs, seq_lengths, supports)
            noisy_features = model.get_features(noisy_inputs, seq_lengths, supports)

            for layer in self.layers_to_penalize:
                if layer in original_features:
                    mse_loss_value = F.mse_loss(original_features[layer], noisy_features[layer]).item()
                    cosine_similarity_value = F.cosine_similarity(
                        original_features[layer].view(original_features[layer].size(0), -1),
                        noisy_features[layer].view(noisy_features[layer].size(0), -1)
                    ).mean().item()

                    # MSE 손실과 코사인 유사도 출력
                    print(f"Layer: {layer}, MSE Loss: {mse_loss_value}")
                    print(f"Cosine Similarity: {cosine_similarity_value}")

                    # 거리 손실 계산 수정 (코사인 유사도를 더하는 방식으로 변경)
                    dist_loss += mse_loss_value + (1 - cosine_similarity_value)

        noise_loss /= self.K
        dist_loss /= self.K

        # Total IRL loss
        total_loss = self.alpha * original_loss + self.beta * noise_loss + self.gamma * dist_loss
        
        # 개별 손실 값 출력
        print(f"Original Loss: {original_loss.item():.4f}")
        print(f"Noise Loss: {noise_loss:.4f}")  
        print(f"Distance Loss: {dist_loss:.4f}")  
        print(f"Total Loss: {total_loss.item():.4f}")

        return total_loss

    
def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, dict):
        return torch.as_tensor(list(x.values())[0])
    else:
        return torch.as_tensor(x)

# 노이즈 함수 정의
def add_noise(inputs):
    # EEG 특화 노이즈 추가 (예: 가우시안 노이즈, 주파수 변조 등)
    return inputs + torch.randn_like(inputs) * 0.5

def train(model_irl, dataloaders, args, device, save_dir, log, tbx):
    """
    Perform training and evaluate on val set
    """
    layers_to_penalize = ['encoder', 'decoder', 'fc']
    
    # Define loss function
    if args.task == 'detection':
        loss_fn=IRLLoss(alpha=0.5, beta=0.5, gamma=1.0, layers_to_penalize=layers_to_penalize).to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)

    # Data loaders
    train_loader = dataloaders['train']
    dev_loader = dataloaders['dev']

    # Get saver
    saver = utils.CheckpointSaver(save_dir,
                                  metric_name=args.metric_name,
                                  maximize_metric=args.maximize_metric,
                                  log=log)

    # To train mode
    model_irl.train()

    # Get optimizer and scheduler
    optimizer = optim.Adam(params=model_irl.parameters(), lr=args.lr_init, weight_decay=args.l2_wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # average meter for validation loss
    nll_meter = utils.AverageMeter()

    # Train
    log.info('Training...')
    epoch = 0
    step = 0
    prev_val_loss = 1e10
    patience_count = 0
    early_stop = False
    # 추가: 건너뛴 배치 수를 추적하기 위한 변수
    skipped_batches = 0

    while (epoch != args.num_epochs) and (not early_stop):
        epoch += 1
        log.info('Starting epoch {}...'.format(epoch))
        total_samples = len(train_loader.dataset)
        print("훈련 토탈 샘플", total_samples)
        with torch.enable_grad(), \
                tqdm(total=total_samples) as progress_bar:
            ### 변경: for 루프 내용을 try-except 블록으로 감싸기
            for batch in train_loader:
                if batch is None or len(batch[0]) != args.train_batch_size:
                    skipped_batches += 1
                    continue
                try:
                    x, y, seq_lengths, supports, _, _ = batch
                    batch_size = x.shape[0]

                    # input seqs
                    x = x.to(device)
                    y = y.view(-1).to(device)  # (batch_size,)
                    seq_lengths = seq_lengths.view(-1).to(device)  # (batch_size,)
                    
                    try:
                        supports = [support.to(device) for support in supports]
                        print("Supports successfully moved to device")
                    except Exception as e:
                        print(f"Error moving supports to device: {e}")
                        # 여기서 continue를 사용하지 않고, 대신 supports를 처리하지 않고 계속 진행
                        pass  
                    

                    ### IRL 적용 GNN 모델 학습 ###
                    print("Starting IRL model training")
                    optimizer.zero_grad()

                    logits = model_irl(x, seq_lengths, supports)
                    logits = to_tensor(logits).view(-1)
                    y = y.view(-1)

                    try:
                        loss = loss_fn(model_irl, x, y, seq_lengths, supports, add_noise)
                        loss_val = loss.item()
                        print(f"Epoch {epoch}, IRLLoss: {loss_val:.4f}")

                        loss.backward()
                        nn.utils.clip_grad_norm_(model_irl.parameters(), args.max_grad_norm)
                        optimizer.step()
                        print(f"Epoch {epoch}, IRLLoss (after backward): {loss.item():.4f}")
                    except Exception as e:
                        print(f"Error in loss calculation: {e}")
                        raise
                    
                    print(f"Loss type: {type(loss)}, Loss value: {loss.item()}")
                    print(f"Learning rate type: {type(optimizer.param_groups[0]['lr'])}, Learning rate value: {optimizer.param_groups[0]['lr']}")
                    
                    # Log info
                    step += batch_size
                    progress_bar.update(batch_size)
                    progress_bar.set_postfix(epoch=epoch,
                                             loss=loss.item(),
                                             lr=optimizer.param_groups[0]['lr'])
                    
                    tbx.add_scalar('train/Loss', loss.item(), step)
                    tbx.add_scalar('train/LR',
                                   optimizer.param_groups[0]['lr'], step)


                ### 추가: 예외 처리 블록
                except Exception as e:
                    print("ERROR",e)
                    skipped_batches += 1
                    continue

            ### 추가: 각 에폭 끝에 건너뛴 배치 수 로깅
            print(f"Epoch {epoch}: Skipped {skipped_batches} batches due to errors")

            if epoch % args.eval_every == 0:
                # Evaluate and save checkpoint
                print('Evaluating at epoch {}...'.format(epoch))
                eval_results = evaluate(model_irl,
                                        dev_loader,
                                        args,
                                        save_dir,
                                        device,
                                        is_test=False,
                                        nll_meter=nll_meter)
                
                if eval_results is None:
                    log.error("Evaluation returned None. Skipping evaluation and checkpoint.")
                    continue  # Skip this epoch or handle the error gracefully

                best_path = saver.save(epoch,
                                       model_irl,
                                       optimizer,
                                       eval_results[args.metric_name])

                # Accumulate patience for early stopping
                if eval_results['loss'] < prev_val_loss :
                    patience_count = 0
                else:
                    patience_count += 1
                prev_val_loss = eval_results['loss']
                

                # Early stop
                if patience_count == args.patience:
                    early_stop = True

                # Back to train mode
                model_irl.train()

                # Log to console
                results_str = ', '.join('{}: {:.3f}'.format(k, v)
                                        for k, v in eval_results.items())
                log.info('Dev {}'.format(results_str))

                # Log to TensorBoard
                log.info('Visualizing in TensorBoard...')
                for k, v in eval_results.items():
                    tbx.add_scalar('eval/{}'.format(k), v, step)

        # Step lr scheduler
        scheduler.step()


def evaluate(
        model_irl,
        dataloader,
        args,
        save_dir,
        device,
        is_test=False,
        nll_meter=None,
        eval_set='dev',
        best_thresh=0.5):
    # To evaluate mode
    model_irl.eval()
    model_irl = model_irl.to('cuda:0')
    layers_to_penalize = ['encoder', 'decoder', 'fc']

    # Define loss function
    if args.task == 'detection':
        loss_fn = IRLLoss(alpha=0.5, beta=0.5, gamma=1.0, layers_to_penalize=layers_to_penalize).to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)


    y_pred_all = []
    y_true_all = []
    y_prob_all = []
    file_name_all = []
    skipped_batches = 0  # 추가: 건너뛴 배치 수 추적
    total_loss = 0
    total_samples = 0
    
    if is_test:
        name="테스트"
    else:
        name = "평가"

    print({name}," dataloader",dataloader)

    #with torch.no_grad(), tqdm(total=len(dataloader.dataset)) as progress_bar:
    with torch.no_grad():
        for batch in tqdm(dataloader):  
            if batch is None or len(batch[0]) == 0:#!= args.test_batch_size:
                skipped_batches += 1
                continue
            try:
                x, y, seq_lengths, supports, _, file_name = batch
                batch_size = x.shape[0]

                 # 디버그: Check target values
                print(f"평가 디버그 Target values: {y.unique()}")


                # Input seqs
                x = x.to(device)
                y = y.view(-1).to(device)  # (batch_size,)
                seq_lengths = seq_lengths.view(-1).to(device)  # (batch_size,)
                #for i in range(len(supports)):
                #    supports[i] = supports[i].to(device)
                try:
                    supports = [support.to(device) for support in supports]
                    print("평가 Supports successfully moved to device")
                except Exception as e:
                    print(f"평가 Error moving supports to device: {e}")
                    # 여기서 continue를 사용하지 않고, 대신 supports를 처리하지 않고 계속 진행
                    pass

                '''
                print("x shape:", x.shape)
                print("seq_lengths shape:", seq_lengths.shape)
                print("supports[0] shape:", supports[0].shape)
                print("supports[0] type:", type(supports[0]))

                print(f"x device: {x.device}")
                print(f"y device: {y.device}")
                print(f"supports device: {supports[0].device}")
                print(f"model device: {next(model_irl.parameters()).device}")
                '''
            
                if args.model_name == "dcrnn":
                    logits = model_irl(x, seq_lengths, supports)
                    print("로짓", logits)
                elif args.model_name == "densecnn":
                    x = x.transpose(-1, -2).reshape(batch_size, -1, args.num_nodes)
                    logits = model_irl(x)
                elif args.model_name == "lstm" or args.model_name == "cnnlstm":
                    logits = model_irl(x, seq_lengths)
                else:
                    raise NotImplementedError
            
                logits = model_irl(x, seq_lengths, supports)
                loss = loss_fn(model_irl, x, y, seq_lengths, supports, add_noise)

                # 디버그: Check model output
                print(f"디버그 {name} Model output shape: {logits.shape}")
                print(f"디버그 {name} Model output values: {logits[:5]}")
                
                if args.num_classes == 1:  # binary detection
                    logits = logits.view(-1)  # (batch_size,)
                    y_prob = torch.sigmoid(logits).cpu().numpy()  # (batch_size, )
                    y_true = y.cpu().numpy().astype(int)
                    y_pred = (y_prob > best_thresh).astype(int)  # (batch_size, )
                else:
                    y_prob = F.softmax(logits, dim=1).cpu().numpy()
                    y_pred = np.argmax(y_prob, axis=1).reshape(-1)  # (batch_size,)
                    y_true = y.cpu().numpy().astype(int)

                 # 디버그 : Check input to loss function
                print(f"디버그 {name} Input to loss function: {logits}")
                print(f"디버그 {name} Target for loss function: {y}")

                #loss = loss_fn(model_irl, x, y, noise_samples)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                if nll_meter is not None:
                    nll_meter.update(loss.item(), batch_size)

                y_pred_all.append(y_pred)
                y_true_all.append(y_true)
                y_prob_all.append(y_prob)
                file_name_all.extend(file_name)

                # Log info
                #progress_bar.update(batch_size)

            except Exception as e:
                skipped_batches += 1
                print("오류!!!!!!!!!!!!!",e)
                continue

    print(f"Evaluation completed. Skipped batches: {skipped_batches}")

    if len(y_pred_all) == 0:
        print("No valid batches processed. Cannot compute results.")
        return None

    y_pred_all = np.concatenate(y_pred_all, axis=0)
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)

    # Threshold search, for detection only
    if (args.task == "detection") and (eval_set == 'dev') and is_test:
        best_thresh = utils.thresh_max_f1(y_true=y_true_all, y_prob=y_prob_all)
        # update dev set y_pred based on best_thresh
        y_pred_all = (y_prob_all > best_thresh).astype(int)  # (batch_size, )
    else:
        best_thresh = best_thresh

    scores_dict, _, _ = utils.eval_dict(y_pred=y_pred_all,
                                        y=y_true_all,
                                        y_prob=y_prob_all,
                                        file_names=file_name_all,
                                        average="binary" if args.task == "detection" else "weighted")


    eval_loss = total_loss / total_samples
    
    results_list = [('loss', eval_loss),
                    ('acc', scores_dict['acc']),
                    ('F1', scores_dict['F1']),
                    ('recall', scores_dict['recall']),
                    ('precision', scores_dict['precision']),
                    ('best_thresh', best_thresh),
                    ('skipped_batches', skipped_batches)] # 추가: 건너뛴 배치 수 결과에 포함
    

    if 'auroc' in scores_dict.keys():
        results_list.append(('auroc', scores_dict['auroc']))
    results = OrderedDict(results_list)

    return results


if __name__ == '__main__':
    main(get_args())
