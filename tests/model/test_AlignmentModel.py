from iatw.IntervalKernels import LinearWarpingKernel
from iatw.Intervals import MinMaxLengthInterval, RelativeLengthInterval
from iatw.jax.loss.intervalkernel.RSS import RSS
from iatw.AlignmentModel import AlignmentModel
from iatw.Parameter import Parameter
from tests.JaxTestCase import JaxTestCase


class TestAlignmentModel(JaxTestCase):

    @staticmethod
    def create_model() -> AlignmentModel:
        def bla(x: float) -> float:
            from math import sin
            return sin(x)

        return AlignmentModel(
            f_ref=bla,
            f_qry=bla,
            ref_begin=Parameter('ref_begin', 0.0, 0.0, 1.0, is_trainable=False),
            qry_begin=Parameter('qry_begin', 0.0, 0.0, 3.14),
            qry_end=Parameter('qry_end', 3.14, 0.0, 5.0))

    def test_init(self):
        self.assertDoesNotRaise(TestAlignmentModel.create_model)
    

    def test_RSS(self):
        m = TestAlignmentModel.create_model()
        m.add_interval(interval=RelativeLengthInterval(model=m, ref_length=0.8))
        
        k = LinearWarpingKernel(interval=list(m.intervals())[0], ref_idx=0)
        r = RSS(kernel=k, weight=1)

        temp = r.evaluate()
        print(temp)
