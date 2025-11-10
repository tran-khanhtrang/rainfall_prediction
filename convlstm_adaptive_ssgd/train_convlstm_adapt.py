import csv
import argparse, os, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from convlstm import ConvLSTM
from adaptive_s_sgd import AdaptiveSSGD
from data_utils import MultiDelayDataset

def prd(y_true, y_pred, eps=1e-12):
    num = torch.sqrt(torch.sum((y_true - y_pred) ** 2))
    den = torch.sqrt(torch.sum((y_true) ** 2) + eps)
    return 100.0 * (num / den)

class ReducerHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        z = self.avgpool(x).squeeze(-1).squeeze(-1)
        return self.fc(z).squeeze(-1)

def run(args):
    # Tăng tốc GPU
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # Dataset & DataLoader
    train_ds = MultiDelayDataset(args.csv, target_col='ANNUAL',
                                 win_len=args.win_len, delays=tuple(args.delays), split='train')
    test_ds  = MultiDelayDataset(args.csv, target_col='ANNUAL',
                                 win_len=args.win_len, delays=tuple(args.delays), split='test')

    pin_mem = (device.type == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=pin_mem)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=pin_mem)

    # Model
    input_dim = len(args.delays)
    model = ConvLSTM(input_dim=input_dim, hidden_dim=args.hidden_dim, kernel_size=3, num_layers=1).to(device)
    head  = ReducerHead(hidden_dim=args.hidden_dim).to(device)

    # Optimizer (AdaptiveSSGD) + loss
    criterion = nn.MSELoss()
    total_steps = args.epochs * max(1, len(train_loader))
    opt = AdaptiveSSGD(list(model.parameters()) + list(head.parameters()),
                       lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                       alpha_max=args.alpha_max, alpha_min=args.alpha_min, total_steps=total_steps)

    best_loss = float('inf')   # theo train loss (để cập nhật leader)
    best_val = float('inf')    # theo val MSE (để lưu best checkpoint)

    def evaluate():
        model.eval(); head.eval()
        mse_sum, prd_sum, n = 0.0, 0.0, 0
        with torch.no_grad():
            for X, y, _ in test_loader:
                X = X.to(device); y = y.to(device)
                seq = [X[:, t] for t in range(X.shape[1])]
                last = model(seq)[-1]
                yhat = head(last)
                mse = criterion(yhat, y)
                mse_sum += mse.item() * y.numel()
                prd_sum += prd(y, yhat).item() * y.numel()
                n += y.numel()
        return mse_sum / max(1, n), prd_sum / max(1, n)

    history = []
    for epoch in range(1, args.epochs+1):
        model.train(); head.train()

        # Thu thập train loss trung bình theo epoch
        epoch_train_loss = 0.0
        cnt = 0

        for X, y, _ in train_loader:
            X = X.to(device); y = y.to(device)

            def closure():
                opt.zero_grad(set_to_none=True)
                seq = [X[:, t] for t in range(X.shape[1])]
                last = model(seq)[-1]
                yhat = head(last)
                loss = criterion(yhat, y)
                loss.backward()
                # chống gradient “đi hoang”
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
                return loss

            loss = opt.step(closure)
            if loss is not None:
                epoch_train_loss += loss.item()
                cnt += 1
                # cập nhật leader theo train loss tốt hơn
                if loss.item() < best_loss - 1e-9:
                    best_loss = loss.item()
                    params_flat = [p.detach().clone().cpu()
                                   for g in opt.param_groups
                                   for p in g['params'] if p.requires_grad]
                    opt.set_leader(params_flat)

        # Đánh giá
        val_mse, val_prd = evaluate()
        train_avg = epoch_train_loss / max(1, cnt)
        history.append({'epoch': epoch, 'train_mse~': train_avg, 'val_mse': val_mse, 'val_prd': val_prd})
        print(f"[Epoch {epoch:03d}] train≈{train_avg:.6f} | val MSE={val_mse:.6f}  PRD={val_prd:.4f}")

        # LR schedule đơn giản
        if epoch in (20, 40, 60):
            for g in opt.param_groups:
                g['lr'] *= 0.3
            print(f"[LR] Decayed to {opt.param_groups[0]['lr']:.2e} at epoch {epoch}")

        # Lưu best checkpoint theo val MSE
        if val_mse < best_val - 1e-12:
            best_val = val_mse
            torch.save({'model': model.state_dict(), 'head': head.state_dict(),
                        'config': vars(args), 'history': history}, 'model_convlstm_adapt_best.pt')
            with open('model_convlstm_adapt_best_metrics.json', 'w') as f:
                json.dump({'val_mse': val_mse, 'val_prd': val_prd, 'epoch': epoch}, f, indent=2)
            print(f"[CKPT] Saved BEST at epoch {epoch}: val MSE={val_mse:.6f}, PRD={val_prd:.4f}")

    # Final evaluate (sau epoch cuối)
    val_mse, val_prd = evaluate()

    # --- Dump predictions (normalized) on test set (robust) ---
    model.eval(); head.eval()
    pred_rows = []

    def _to_int_list(obj, batch):
        # Trả về list[int] độ dài = batch (nếu không suy được thì None)
        import torch as _torch
        if isinstance(obj, (int, float, str)):
            try: return [int(obj)] * batch
            except: return [None] * batch
        if _torch.is_tensor(obj):
            if obj.ndim == 0: return [int(obj.item())] * batch
            obj = obj.detach().cpu().view(-1).tolist()
            return [int(x) for x in obj]
        if isinstance(obj, list):
            out = []
            for m in obj:
                if isinstance(m, dict) and 'year_pred' in m:
                    out.append(int(m['year_pred']))
                elif _torch.is_tensor(m):
                    out.append(int(m.item() if m.ndim==0 else m.view(-1)[0].item()))
                else:
                    try: out.append(int(m))
                    except: out.append(None)
            return out
        if isinstance(obj, dict):
            if 'year_pred' in obj:
                return _to_int_list(obj['year_pred'], batch)
            for v in obj.values():
                return _to_int_list(v, batch)
            return [None] * batch
        return [None] * batch

    with torch.no_grad():
        for X, y, meta in test_loader:
            X = X.to(device); y = y.to(device)
            seq = [X[:, t] for t in range(X.shape[1])]
            last = model(seq)[-1]
            yhat = head(last)
            years = _to_int_list(meta, y.shape[0])
            for yi, ypi, yr in zip(y.cpu().tolist(), yhat.cpu().tolist(), years):
                pred_rows.append([yr, yi, ypi])

    out_pred = args.out.replace('.pt','_preds_test_norm.csv')
    with open(out_pred, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['year','y_true_norm','y_pred_norm'])
        w.writerows(pred_rows)
    print(f"[DONE] Wrote preds (normalized) -> {out_pred}")

    # Lưu checkpoint “cuối cùng”
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    torch.save({'model': model.state_dict(), 'head': head.state_dict(),
                'config': vars(args), 'history': history}, args.out)
    with open(args.out.replace('.pt', '_metrics.json'), 'w') as f:
        json.dump({'val_mse': val_mse, 'val_prd': val_prd, 'history': history}, f, indent=2)
    print(f"[DONE] Saved model to {args.out}")
    print(f"[DONE] MSE={val_mse:.6f}  PRD={val_prd:.4f}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, required=True)
    p.add_argument('--win_len', type=int, default=5)
    p.add_argument('--delays', type=int, nargs='+', default=[1,2,3])
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--momentum', type=float, default=0.0)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--alpha_max', type=float, default=0.3)
    p.add_argument('--alpha_min', type=float, default=0.0)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--out', type=str, default='model_convlstm_adapt.pt')
    args = p.parse_args()
    run(args)
