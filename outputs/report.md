## run `b0_288`

- args: `{"arch": "efficientnet_b0", "size": 288, "folds": 5, "max_folds": 0, "head_epochs": 4, "ft_epochs": 46, "batch_size": 32, "lr": 0.0002, "mixup": 0.0, "no_crop": false, "tag": "b0_288", "seed": 42}`
- OOF accuracy: **0.9140**, macro F1: **0.9132**
- folds: acc=0.9200/f1=0.9195/3.9min, acc=0.8800/f1=0.8771/4.0min, acc=0.9500/f1=0.9490/3.9min, acc=0.9000/f1=0.9019/3.9min, acc=0.9200/f1=0.9157/3.8min
## run `cnx320`

- args: `{"arch": "convnext_tiny.fb_in22k_ft_in1k", "size": 320, "folds": 5, "max_folds": 0, "head_epochs": 4, "ft_epochs": 40, "batch_size": 32, "lr": 0.0002, "weight_decay": 0.05, "label_smoothing": 0.1, "mixup": 0.2, "cutmix": 1.0, "ema": 0.0, "raug": true, "raug_ops": 2, "raug_mag": 9, "no_crop": false, "mask_bg": false, "tag": "cnx320", "seed": 42, "subset": "", "resume": true, "pseudo": "", "pseudo_thresh": 0.9}`
- OOF accuracy: **0.9180**, macro F1: **0.9179**
- folds: acc=0.9000/f1=0.9010/0.5min, acc=0.9200/f1=0.9182/0.4min, acc=0.9300/f1=0.9287/0.4min, acc=0.9200/f1=0.9203/0.4min, acc=0.9200/f1=0.9198/0.4min
## run `cnx320p`

- args: `{"arch": "convnext_tiny.fb_in22k_ft_in1k", "size": 320, "folds": 5, "max_folds": 0, "head_epochs": 4, "ft_epochs": 40, "batch_size": 32, "lr": 0.0002, "weight_decay": 0.05, "label_smoothing": 0.1, "mixup": 0.2, "cutmix": 1.0, "ema": 0.0, "raug": true, "raug_ops": 2, "raug_mag": 9, "no_crop": false, "mask_bg": false, "tag": "cnx320p", "seed": 42, "subset": "", "resume": false, "pseudo": "D:\\机器学习\\2026task1\\outputs\\test_probs.npz", "pseudo_thresh": 0.9}`
- OOF accuracy: **0.9280**, macro F1: **0.9280**
- folds: acc=0.9500/f1=0.9505/4.7min, acc=0.9300/f1=0.9286/4.9min, acc=0.9200/f1=0.9200/4.8min, acc=0.9300/f1=0.9307/4.7min, acc=0.9100/f1=0.9077/4.6min
## run `cnx320m`

- args: `{"arch": "convnext_tiny.fb_in22k_ft_in1k", "size": 320, "folds": 5, "max_folds": 0, "head_epochs": 4, "ft_epochs": 40, "batch_size": 32, "lr": 0.0002, "weight_decay": 0.05, "label_smoothing": 0.1, "mixup": 0.2, "cutmix": 1.0, "ema": 0.0, "raug": true, "raug_ops": 2, "raug_mag": 9, "no_crop": false, "mask_bg": true, "tag": "cnx320m", "seed": 42, "subset": "", "resume": false, "pseudo": "D:\\机器学习\\2026task1\\outputs\\test_probs.npz", "pseudo_thresh": 0.9}`
- OOF accuracy: **0.9140**, macro F1: **0.9134**
- folds: acc=0.9200/f1=0.9200/5.9min, acc=0.9000/f1=0.8999/5.5min, acc=0.9500/f1=0.9497/5.5min, acc=0.9000/f1=0.8977/5.4min, acc=0.9000/f1=0.8953/5.4min
## run `cnx384m`

- args: `{"arch": "convnext_tiny.fb_in22k_ft_in1k", "size": 384, "folds": 5, "max_folds": 0, "head_epochs": 4, "ft_epochs": 40, "batch_size": 32, "lr": 0.0002, "weight_decay": 0.05, "label_smoothing": 0.1, "mixup": 0.2, "cutmix": 1.0, "ema": 0.0, "raug": true, "raug_ops": 2, "raug_mag": 9, "no_crop": false, "mask_bg": true, "tag": "cnx384m", "seed": 42, "subset": "", "resume": false, "pseudo": "D:\\机器学习\\2026task1\\outputs\\test_probs.npz", "pseudo_thresh": 0.9}`
- OOF accuracy: **0.9240**, macro F1: **0.9238**
- folds: acc=0.9300/f1=0.9307/8.1min, acc=0.9000/f1=0.8995/8.4min, acc=0.9600/f1=0.9599/7.9min, acc=0.9100/f1=0.9086/9.4min, acc=0.9200/f1=0.9161/7.8min

## ensemble of out-of-fold predictions

- accuracy: **0.9380**, macro F1: **0.9380**
- expected errors on 378 test images: **23.4**

```
                   precision    recall  f1-score   support

      Black-grass     0.8302    0.8800    0.8544       100
     Common wheat     0.9700    0.9700    0.9700       100
 Loose Silky-bent     0.9032    0.8400    0.8705       100
Scentless Mayweed     1.0000    1.0000    1.0000       100
       Sugar beet     0.9901    1.0000    0.9950       100

         accuracy                         0.9380       500
        macro avg     0.9387    0.9380    0.9380       500
     weighted avg     0.9387    0.9380    0.9380       500
```
