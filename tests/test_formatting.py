import unittest
from urban_transfer.formatting import metric_text,epoch_text

class ThreeDecimalFormattingTests(unittest.TestCase):
    def test_metrics(self):
        self.assertEqual(metric_text({'n':85808,'mse':13.458240611597061,'mae':3.188526008023801,'r2':-3.6163255895649273}),
            'n=85808 mse=13.458 mae=3.189 r2=-3.616')
    def test_null_r2(self):
        self.assertEqual(metric_text({'n':1,'mse':0.123456,'mae':0.123456,'r2':None}),
            'n=1 mse=0.123 mae=0.123 r2=N/A')
    def test_epoch(self):
        record={'epoch':18,'train':{'mean_mse':1.09663245},'validation':{'overall':{'mse':0.961612345}},'best_val_mse':0.961612345}
        self.assertEqual(epoch_text(record),'[train] epoch=18 train_mse=1.097 val_mse=0.962 best_val_mse=0.962')

if __name__=='__main__':unittest.main()
