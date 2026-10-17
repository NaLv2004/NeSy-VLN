import unittest
import torch
from nesy_vla.visual_transfer import single_image_features


class VisualTransferTests(unittest.TestCase):
    def test_repeated_rgb_equivalence_and_no_heads(self):
        torch.manual_seed(10)
        weight=torch.randn(8,9,4,4);bias=torch.randn(8);rgb=torch.randn(2,3,24,24)
        state={'encoder.features.0.0.weight':weight,'encoder.features.0.0.bias':bias,
               'cost_head.weight':torch.randn(17,8),'category.weight':torch.randn(2,16)}
        result=single_image_features(state)
        old=torch.nn.functional.conv2d(rgb.repeat(1,3,1,1),weight,bias,stride=4)
        new=torch.nn.functional.conv2d(rgb,result['0.0.weight'],bias,stride=4)
        torch.testing.assert_close(old,new,atol=2e-5,rtol=1e-5)
        self.assertEqual(set(result),{'0.0.weight','0.0.bias'})
    def test_reject_unknown_input_architecture(self):
        with self.assertRaises(ValueError):single_image_features({'encoder.features.0.0.weight':torch.randn(8,6,4,4)})
    def test_single_rgb_is_bit_exact(self):
        weight=torch.randn(8,3,4,4)
        self.assertTrue(torch.equal(weight,single_image_features({'encoder.features.0.0.weight':weight})['0.0.weight']))


if __name__=='__main__':unittest.main()
