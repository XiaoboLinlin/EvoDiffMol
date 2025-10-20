import argparse
import shutil

import torch.distributed as dist
import torch.utils.tensorboard
import yaml
from easydict import EasyDict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader
from tqdm.auto import tqdm

from configs.datasets_config import get_dataset_info
from evodiffmol.models.epsnet import get_model
from datasets.qm9.utils import prepare_context, compute_mean_mad
from evodiffmol.utils.common import get_optimizer, get_scheduler
from evodiffmol.utils.datasets import QM93D, QM403D, Geom, General3D, General3DOnDemand
from evodiffmol.utils.dataset_scaffold import QM40ScaffoldDataset
from evodiffmol.utils.misc import *
from evodiffmol.utils.transforms import *
from evodiffmol.utils.training_utils import train_epoch, validate_epoch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=None,
                    help='qm9, qm40, geom, guacamol, moses (auto-detected from config if not specified)')
parser.add_argument('--config', type=str, required=True,
                    help='Path to YAML config file (no directories supported)')
parser.add_argument('--resume_checkpoint', type=str, default=None,
                    help='Path to checkpoint file (.pt) to resume from')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--use_mixed_precision', type=bool, default=False)
parser.add_argument('--dp', type=bool, default=True)
parser.add_argument('--resume_iter', type=int, default=None)
parser.add_argument('--logdir', type=str, required=True,
                    help='Directory to save logs and checkpoints')
parser.add_argument("--context", nargs='+', default=[],
                    help='arguments : homo | lumo | alpha | gap | mu | Cv')
parser.add_argument('--remove_h', action='store_true', default=False,
                    help='Remove hydrogen atoms from dataset')

# Scaffold training arguments
parser.add_argument('--scaffold_mol2', type=str, default=None,
                    help='Path to MOL2 file containing scaffold structure. If provided, enables scaffold training mode.')
parser.add_argument('--min_scaffold_molecules', type=int, default=200,
                    help='Minimum number of scaffold molecules required')
parser.add_argument('--max_scaffold_molecules', type=int, default=10000,
                    help='Maximum number of scaffold molecules to use')


def train(it):
    """Train function using the refactored training utilities."""
    train_metrics = train_epoch(
        model=model,
        train_loader=train_loader,
        optimizer_global=optimizer_global,
        optimizer_local=optimizer_local,
        config=config,
        device=device,
        writer=writer,
        logger=logger,
        iteration=it,
        property_norms=property_norms,
        context_args=args.context,
        ema=None
    )
    return train_metrics


def validate(it):
    """Validate function using the refactored training utilities."""
    avg_val_loss = validate_epoch(
        model=model,
        val_loader=val_loader,
        config=config,
        device=device,
        scheduler_global=scheduler_global,
        scheduler_local=scheduler_local,
        writer=writer,
        logger=logger,
        iteration=it,
        property_norms=property_norms_val,
        context_args=args.context,
        ema=None
    )
    return avg_val_loss


# ------------------------------------------------------------------------------
# Training file  not in ddp mode
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    # Auto-detect dataset and remove_h from config filename if not explicitly set
    if args.dataset is None:
        if 'qm9' in args.config:
            args.dataset = 'qm9'
        elif 'qm40' in args.config:
            args.dataset = 'qm40'
        elif 'guacamol' in args.config:
            args.dataset = 'guacamol'
        elif 'moses' in args.config:
            args.dataset = 'moses'
        else:
            args.dataset = 'geom'
    
    # Auto-detect remove_h from config filename
    if 'without_h' in args.config:
        args.remove_h = True

    # Validate config is a YAML file, not a directory
    if not args.config.endswith('.yml') and not args.config.endswith('.yaml'):
        raise ValueError("--config must be a YAML file (.yml or .yaml), not a directory")
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    config_path = args.config
    resume = args.resume_checkpoint is not None
    
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    # config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    config_name = '%s_full_ddpm_2losses' % args.dataset  # 'qm9_full_temb_charge_norm_edmdataset' # log name
    seed_all(config.train.seed)

    if resume:
        if not os.path.exists(args.resume_checkpoint):
            raise FileNotFoundError(f"Checkpoint file not found: {args.resume_checkpoint}")
        log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag='resume')
    else:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name)
        if not os.path.exists(os.path.join(log_dir, 'models')):
            shutil.copytree('./models', os.path.join(log_dir, 'models'), dirs_exist_ok=True)

    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)

    # Logging
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))
    shutil.copyfile('./train.py', os.path.join(log_dir, 'train.py'))
    
    if resume:
        logger.info(f"Resuming from checkpoint: {args.resume_checkpoint}")
    # Datasets and loaders
    logger.info('Loading %s datasets...' % (args.dataset))

    dataset_info = get_dataset_info(args.dataset, remove_h=args.remove_h)
    transforms = Compose([CountNodesPerGraph(), GetAdj(), AtomFeat(dataset_info['atom_index'])])

    # Handle scaffold training mode
    scaffold_training = args.scaffold_mol2 is not None
    
    if scaffold_training:
        if args.dataset != 'qm40':
            raise ValueError("Scaffold training currently only supports qm40 dataset")
        
        logger.info(f'Loading scaffold-filtered dataset with MOL2: {args.scaffold_mol2}')
        train_set = QM40ScaffoldDataset(
            scaffold_mol2_path=args.scaffold_mol2,
            root='./ga_output',
            split='train',
            min_molecules=args.min_scaffold_molecules,
            max_molecules=args.max_scaffold_molecules,
            remove_h=args.remove_h,
            pre_transform=transforms
        )
        val_set = QM40ScaffoldDataset(
            scaffold_mol2_path=args.scaffold_mol2,
            root='./ga_output',
            split='valid',
            min_molecules=max(50, args.min_scaffold_molecules // 4),
            max_molecules=args.max_scaffold_molecules // 4,
            remove_h=args.remove_h,
            pre_transform=transforms
        )
        logger.info(f'Scaffold training: {len(train_set)} train, {len(val_set)} validation molecules')
    else:
        # Regular training mode
        if args.dataset == 'qm9':
            train_set = QM93D('train', pre_transform=transforms)
            val_set = QM93D('valid', pre_transform=transforms)
        elif args.dataset == 'qm40':
            train_set = QM403D('train',pre_transform=transforms, remove_h=args.remove_h)
            val_set = QM403D('valid', pre_transform=transforms, remove_h=args.remove_h)
        elif args.dataset == 'geom':
            train_set = Geom(pre_transform=transforms)
            val_set = None  # Geom doesn't have separate validation
        elif args.dataset in ['guacamol', 'moses']:
            # Choose dataset class based on size and memory requirements
            # MOSES with_h is very large (~1.58M molecules) and needs on-demand loading
            # GuacaMol and MOSES without_h can use InMemoryDataset
            
            if args.dataset == 'moses' and not args.remove_h:
                # MOSES with_h: Use on-demand loading to avoid memory explosion
                logger.info("Using General3DOnDemand for MOSES with_h (memory-efficient)")
                train_set = General3DOnDemand(args.dataset, 'train', transform=transforms, remove_h=args.remove_h)
                val_set = General3DOnDemand(args.dataset, 'valid', transform=transforms, remove_h=args.remove_h)
            else:
                # MOSES without_h, GuacaMol: Use standard InMemoryDataset
                logger.info(f"Using General3D (InMemoryDataset) for {args.dataset} (remove_h={args.remove_h})")
                train_set = General3D(args.dataset, 'train', pre_transform=transforms, remove_h=args.remove_h)
                val_set = General3D(args.dataset, 'valid', pre_transform=transforms, remove_h=args.remove_h)
        else:
            raise Exception(f'Wrong dataset name: {args.dataset}. Supported: qm9, qm40, geom, guacamol, moses')

    train_loader = DataLoader(train_set, config.train.batch_size, shuffle=True)
    if val_set is not None:
        val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False)

    # if context
    # args.context = ['alpha']
    if len(args.context) > 0:
        if args.dataset in ['guacamol', 'moses']:
            raise ValueError(f"Context conditioning not supported for {args.dataset} dataset (no property targets)")
        print(f'Conditioning on {args.context}')
        property_norms = compute_mean_mad(train_set, args.context, args.dataset)
        if args.dataset in ['qm9', 'qm40']:  # QM40 also has validation set
            property_norms_val = compute_mean_mad(val_set, args.context, args.dataset)
        else:
            property_norms_val = property_norms  # Use train norms for datasets without val set
    else:
        property_norms = None
        property_norms_val = None
        context = None

    # Model
    logger.info('Building model...')
    config.model.context = args.context
    # Auto-calculate num_atom from dataset configuration (num_atom_types + 1 for padding)
    config.model.num_atom = len(dataset_info['atom_decoder']) + 1
    logger.info(f'Dataset: {args.dataset}, remove_h: {args.remove_h}, num_atom: {config.model.num_atom}')

    model = get_model(config.model).to(device)

    # Optimizer
    optimizer_global = get_optimizer(config.train.optimizer, model.model_global)
    scheduler_global = get_scheduler(config.train.scheduler, optimizer_global)
    optimizer_local = get_optimizer(config.train.optimizer, model.model_local)
    scheduler_local = get_scheduler(config.train.scheduler, optimizer_local)

    start_iter = 0

    # Resume from checkpoint
    if resume:
        ckpt_path = args.resume_checkpoint
        logger.info('Resuming from: %s' % ckpt_path)
        ckpt = torch.load(ckpt_path, weights_only=False)
        
        # Extract iteration from checkpoint
        start_iter = ckpt.get('iteration', 0)
        if args.resume_iter is not None:
            start_iter = args.resume_iter
        logger.info('Iteration: %d' % start_iter)
        
        model.load_state_dict(ckpt['model'])
        optimizer_global.load_state_dict(ckpt['optimizer_global'])
        optimizer_local.load_state_dict(ckpt['optimizer_local'])
        scheduler_global.load_state_dict(ckpt['scheduler_global'])
        scheduler_local.load_state_dict(ckpt['scheduler_local'])

    # No EMA for scaffold training

    best_val_loss = float('inf')
    for it in range(start_iter, config.train.max_iters + 1):
        start_time = time.time()
        train(it)
        end_time = (time.time() - start_time)
        print('each iteration requires {} s'.format(end_time))
        avg_val_loss = validate(it)
        if it % config.train.val_freq == 0:
            if avg_val_loss < best_val_loss:
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer_global': optimizer_global.state_dict(),
                    'scheduler_global': scheduler_global.state_dict(),
                    'optimizer_local': optimizer_local.state_dict(),
                    'scheduler_local': scheduler_local.state_dict(),
                    'iteration': it,
                    'avg_val_loss': avg_val_loss,
                }, ckpt_path)
                print('Successfully saved the model!')
                best_val_loss = avg_val_loss

    logger.info("Training completed!")
    if scaffold_training:
        print(f"\n🎯 Scaffold training finished! Use trained model for genetic algorithm optimization.")
