package ilpInference;

import java.util.Map;
import java.util.logging.Logger;

import net.sf.javailp.Constraint;
import net.sf.javailp.Linear;
import net.sf.javailp.OptType;
import net.sf.javailp.Problem;
import net.sf.javailp.Result;
import net.sf.javailp.Solver;
import net.sf.javailp.SolverFactory;
import net.sf.javailp.SolverFactoryGLPK;
import net.sf.javailp.SolverFactoryLpSolve;

public class TestJavaILP {
    static Logger logger = Logger.getLogger(TestJavaILP.class.getName());
    
    static String objectiveString = "-166860.0000001519*z0_y0 + -166856.0000001519*z0_y1 + -166858.66666681864*z0_y2 + -166866.16666681852*z0_y3 + -166862.16666681864*z0_y4 + -166861.00000015195*z0_y5 + -166855.66666681858*z0_y6 + -166861.33333348524*z0_y7 + -166864.3333334853*z0_y8 + -166849.83333348527*z0_y9 + -166842.1666668185*z0_y10 + -166836.00000015207*z0_y11 + -166871.6666668185*z0_y12 + -166855.16666681867*z0_y13 + -166858.8333334853*z0_y14 + -166884.16666681852*z0_y15 + -166858.8333334852*z0_y17 + -166862.6666668185*z0_y16 + -166852.50000015192*z0_y19 + -166860.1666668186*z0_y18 + -166853.16666681855*z0_y21 + -166854.83333348518*z0_y20 + -166862.50000015195*z0_y23 + -166849.66666681852*z0_y22 + -166856.1666668186*z0_y25 + -166861.83333348512*z0_y24 + -166854.33333348524*z0_y27 + -166862.16666681855*z0_y26 + -166861.6666668186*z0_y29 + -166859.33333348518*z0_y28 + -166860.3333334853*z0_y31 + -166857.00000015192*z0_y30 + -166851.00000015192*z0_y34 + -166861.6666668186*z0_y35 + -166861.1666668185*z0_y32 + -166855.8333334852*z0_y33 + -166856.66666681855*z0_y38 + -166859.3333334852*z0_y39 + -166859.66666681855*z0_y36 + -166861.50000015186*z0_y37 + -166855.8333334852*z0_y40 + -166864.16666681867*z0_y41 + -181813.33333348556*z1_y0 + -181813.16666681896*z1_y1 + -181817.83333348567*z1_y2 + -181817.16666681893*z1_y3 + -181828.50000015236*z1_y4 + -181817.166666819*z1_y5 + -181834.33333348567*z1_y6 + -181800.16666681893*z1_y7 + -181822.1666668189*z1_y8 + -181813.16666681893*z1_y9 + -181813.50000015224*z1_y10 + -181790.666666819*z1_y11 + -181846.16666681887*z1_y12 + -181815.33333348559*z1_y13 + -181815.3333334857*z1_y14 + -181854.6666668189*z1_y15 + -181826.66666681887*z1_y17 + -181815.00000015224*z1_y16 + -181810.0000001523*z1_y19 + -181820.166666819*z1_y18 + -181807.16666681887*z1_y21 + -181809.00000015224*z1_y20 + -181821.50000015233*z1_y23 + -181813.1666668188*z1_y22 + -181815.66666681887*z1_y25 + -181822.33333348553*z1_y24 + -181810.1666668189*z1_y27 + -181830.33333348556*z1_y26 + -181820.16666681896*z1_y29 + -181813.33333348553*z1_y28 + -181811.50000015236*z1_y31 + -181821.50000015233*z1_y30 + -181813.00000015227*z1_y34 + -181815.166666819*z1_y35 + -181816.83333348544*z1_y32 + -181822.00000015218*z1_y33 + -181821.00000015227*z1_y38 + -181815.00000015224*z1_y39 + -181818.50000015218*z1_y36 + -181822.00000015218*z1_y37 + -181813.83333348559*z1_y40 + -181820.33333348567*z1_y41 + -181756.66666681887*z2_y0 + -181747.5000001523*z2_y1 + -181751.0000001524*z2_y2 + -181754.83333348559*z2_y3 + -181766.16666681904*z2_y4 + -181752.66666681904*z2_y5 + -181764.00000015233*z2_y6 + -181735.83333348567*z2_y7 + -181764.50000015227*z2_y8 + -181747.00000015227*z2_y9 + -181742.1666668189*z2_y10 + -181710.83333348564*z2_y11 + -181785.83333348556*z2_y12 + -181754.83333348564*z2_y13 + -181751.00000015236*z2_y14 + -181789.50000015224*z2_y15 + -181763.6666668189*z2_y17 + -181748.1666668189*z2_y16 + -181745.166666819*z2_y19 + -181752.666666819*z2_y18 + -181748.5000001522*z2_y21 + -181743.16666681893*z2_y20 + -181755.666666819*z2_y23 + -181747.66666681887*z2_y22 + -181748.66666681887*z2_y25 + -181758.16666681887*z2_y24 + -181743.83333348553*z2_y27 + -181756.50000015227*z2_y26 + -181751.50000015233*z2_y29 + -181748.0000001522*z2_y28 + -181744.16666681902*z2_y31 + -181755.00000015233*z2_y30 + -181748.33333348564*z2_y34 + -181751.00000015233*z2_y35 + -181749.50000015215*z2_y32 + -181759.50000015218*z2_y33 + -181754.16666681896*z2_y38 + -181748.3333334856*z2_y39 + -181753.83333348559*z2_y36 + -181756.1666668189*z2_y37 + -181752.50000015227*z2_y40 + -181756.83333348567*z2_y41 + -155253.83333348276*z3_y0 + -155263.00000014945*z3_y1 + -155269.33333348285*z3_y2 + -155269.5000001494*z3_y3 + -155254.33333348276*z3_y4 + -155257.83333348276*z3_y5 + -155259.33333348273*z3_y6 + -155256.83333348276*z3_y7 + -155258.66666681613*z3_y8 + -155251.83333348282*z3_y9 + -155247.50000014933*z3_y10 + -155232.16666681625*z3_y11 + -155277.00000014936*z3_y12 + -155280.6666668161*z3_y13 + -155258.8333334828*z3_y14 + -155286.3333334827*z3_y15 + -155262.5000001494*z3_y17 + -155268.16666681602*z3_y16 + -155267.33333348282*z3_y19 + -155260.83333348273*z3_y18 + -155251.00000014942*z3_y21 + -155256.5000001494*z3_y20 + -155282.00000014948*z3_y23 + -155251.33333348267*z3_y22 + -155256.16666681608*z3_y25 + -155262.83333348267*z3_y24 + -155261.50000014942*z3_y27 + -155265.83333348273*z3_y26 + -155265.50000014945*z3_y29 + -155264.50000014942*z3_y28 + -155261.16666681613*z3_y31 + -155259.83333348276*z3_y30 + -155249.6666668161*z3_y34 + -155263.33333348276*z3_y35 + -155254.50000014936*z3_y32 + -155259.00000014936*z3_y33 + -155261.83333348273*z3_y38 + -155261.6666668161*z3_y39 + -155260.50000014936*z3_y36 + -155263.33333348276*z3_y37 + -155257.83333348273*z3_y40 + -155269.50000014945*z3_y41 + -339727.1666669735*z4_y0 + -339731.16666697344*z4_y1 + -339735.6666669737*z4_y2 + -339731.33333364*z4_y3 + -339738.8333336403*z4_y4 + -339730.1666669735*z4_y5 + -339736.1666669734*z4_y6 + -339738.16666697344*z4_y7 + -339747.8333336403*z4_y8 + -339728.6666669735*z4_y9 + -339710.33333363984*z4_y10 + -339699.33333364024*z4_y11 + -339749.66666697327*z4_y12 + -339732.83333364*z4_y13 + -339735.16666697356*z4_y14 + -339778.6666669733*z4_y15 + -339737.6666669734*z4_y17 + -339737.0000003067*z4_y16 + -339725.8333336399*z4_y19 + -339735.8333336401*z4_y18 + -339716.0000003066*z4_y21 + -339726.1666669733*z4_y20 + -339739.66666697356*z4_y23 + -339726.16666697327*z4_y22 + -339729.3333336401*z4_y25 + -339735.16666697303*z4_y24 + -339729.0000003068*z4_y27 + -339733.1666669734*z4_y26 + -339735.50000030675*z4_y29 + -339725.1666669733*z4_y28 + -339730.50000030687*z4_y31 + -339732.1666669735*z4_y30 + -339718.1666669734*z4_y34 + -339740.0000003068*z4_y35 + -339729.16666697327*z4_y32 + -339729.16666697327*z4_y33 + -339729.66666697327*z4_y38 + -339732.0000003067*z4_y39 + -339732.00000030664*z4_y36 + -339739.1666669733*z4_y37 + -339729.00000030664*z4_y40 + -339738.5000003067*z4_y41 + -214939.00000019852*z5_y0 + -214944.33333353192*z5_y1 + -214950.33333353195*z5_y2 + -214960.0000001986*z5_y3 + -214953.33333353195*z5_y4 + -214942.66666686526*z5_y5 + -214950.3333335319*z5_y6 + -214938.83333353192*z5_y7 + -214952.00000019863*z5_y8 + -214930.50000019863*z5_y9 + -214932.5000001985*z5_y10 + -214924.16666686547*z5_y11 + -214950.33333353183*z5_y12 + -214961.00000019872*z5_y13 + -214944.5000001986*z5_y14 + -214970.83333353183*z5_y15 + -214934.50000019858*z5_y17 + -214956.33333353183*z5_y16 + -214937.33333353203*z5_y19 + -214944.16666686526*z5_y18 + -214933.00000019855*z5_y21 + -214942.33333353186*z5_y20 + -214942.83333353195*z5_y23 + -214944.5000001985*z5_y22 + -214938.16666686526*z5_y25 + -214956.83333353186*z5_y24 + -214949.5000001986*z5_y27 + -214956.3333335319*z5_y26 + -214955.83333353195*z5_y29 + -214946.66666686517*z5_y28 + -214947.66666686538*z5_y31 + -214945.50000019858*z5_y30 + -214940.00000019855*z5_y34 + -214949.00000019863*z5_y35 + -214952.8333335318*z5_y32 + -214941.00000019846*z5_y33 + -214948.66666686526*z5_y38 + -214948.33333353192*z5_y39 + -214949.6666668652*z5_y36 + -214953.50000019858*z5_y37 + -214942.1666668653*z5_y40 + -214957.16666686532*z5_y41 + -206464.00000019852*z6_y0 + -206461.83333353192*z6_y1 + -206465.6666668653*z6_y2 + -206469.16666686515*z6_y3 + -206464.50000019858*z6_y4 + -206464.1666668652*z6_y5 + -206466.00000019855*z6_y6 + -206465.00000019858*z6_y7 + -206463.33333353192*z6_y8 + -206461.33333353195*z6_y9 + -206443.66666686512*z6_y10 + -206437.50000019875*z6_y11 + -206489.16666686515*z6_y12 + -206458.1666668653*z6_y13 + -206460.66666686535*z6_y14 + -206504.83333353183*z6_y15 + -206467.66666686517*z6_y17 + -206468.66666686517*z6_y16 + -206456.00000019863*z6_y19 + -206463.66666686523*z6_y18 + -206458.83333353183*z6_y21 + -206461.6666668652*z6_y20 + -206462.16666686526*z6_y23 + -206453.16666686515*z6_y22 + -206458.33333353186*z6_y25 + -206466.3333335318*z6_y24 + -206474.50000019855*z6_y27 + -206478.6666668652*z6_y26 + -206463.83333353192*z6_y29 + -206463.33333353186*z6_y28 + -206464.00000019866*z6_y31 + -206459.83333353192*z6_y30 + -206463.16666686526*z6_y34 + -206464.16666686523*z6_y35 + -206466.50000019846*z6_y32 + -206462.50000019852*z6_y33 + -206462.83333353192*z6_y38 + -206461.3333335319*z6_y39 + -206464.8333335318*z6_y36 + -206472.16666686517*z6_y37 + -206462.66666686517*z6_y40 + -206469.16666686532*z6_y41 + -251173.83333358177*z7_y0 + -251172.8333335818*z7_y1 + -251177.3333335819*z7_y2 + -251173.6666669151*z7_y3 + -251172.00000024855*z7_y4 + -251179.00000024852*z7_y5 + -251175.83333358177*z7_y6 + -251171.00000024852*z7_y7 + -251177.00000024852*z7_y8 + -251167.16666691517*z7_y9 + -251155.666666915*z7_y10 + -251153.0000002487*z7_y11 + -251203.8333335817*z7_y12 + -251173.83333358192*z7_y13 + -251172.16666691526*z7_y14 + -251199.33333358174*z7_y15 + -251172.1666669151*z7_y17 + -251174.6666669151*z7_y16 + -251169.50000024852*z7_y19 + -251171.50000024846*z7_y18 + -251163.1666669151*z7_y21 + -251175.16666691512*z7_y20 + -251176.00000024852*z7_y23 + -251164.50000024837*z7_y22 + -251169.16666691515*z7_y25 + -251175.83333358166*z7_y24 + -251164.0000002485*z7_y27 + -251178.66666691512*z7_y26 + -251176.33333358186*z7_y29 + -251174.33333358177*z7_y28 + -251174.50000024858*z7_y31 + -251170.00000024852*z7_y30 + -251166.3333335818*z7_y34 + -251176.6666669152*z7_y35 + -251171.33333358166*z7_y32 + -251174.83333358172*z7_y33 + -251176.33333358183*z7_y38 + -251175.3333335818*z7_y39 + -251173.66666691506*z7_y36 + -251182.66666691512*z7_y37 + -251171.0000002484*z7_y40 + -251183.16666691526*z7_y41 + -7.33333333333394*y0 + -4.500000000001819*y1 + -7.500000000001819*y2 + -9.83333333333394*y3 + -17.83333333333394*y4 + -5.66666666666606*y5 + -10.83333333333394*y6 + -12.500000000005457*y7 + -7.666666666667879*y8 + -13.666666666669698*y9 + -11.83333333333394*y10 + -27.666666666675155*y11 + -19.833333333344854*y12 + -20.500000000003638*y13 + -11.000000000001819*y14 + -19.1666666666697*y15 + -4.5*y16 + -15.333333333339397*y17 + -4.66666666666606*y18 + -7.833333333330302*y19 + -10.333333333335759*y20 + -11.500000000005457*y21 + -14.333333333335759*y22 + -13.166666666669698*y23 + -6.166666666669698*y24 + -3.999999999998181*y25 + -4.833333333330302*y26 + -9.0*y27 + -1.499999999996362*y28 + -6.16666666666606*y29 + -2.999999999998181*y30 + -4.333333333332121*y31 + -5.5*y32 + -7.166666666664241*y33 + -9.5*y34 + -6.16666666666606*y35 + 1.8189894035458565E-12*y36 + -0.8333333333321207*y37 + -0.999999999998181*y38 + -1.999999999998181*y39 + -1.1666666666642413*y40 + 0.0*y41";
    static String constraintsString = "1*z0_y0 + 1*z0_y1 + 1*z0_y2 + 1*z0_y3 + 1*z0_y4 + 1*z0_y5 + 1*z0_y6 + 1*z0_y7 + 1*z0_y8 + 1*z0_y9 + 1*z0_y10 + 1*z0_y11 + 1*z0_y12 + 1*z0_y13 + 1*z0_y14 + 1*z0_y15 + 1*z0_y16 + 1*z0_y17 + 1*z0_y18 + 1*z0_y19 + 1*z0_y20 + 1*z0_y21 + 1*z0_y22 + 1*z0_y23 + 1*z0_y24 + 1*z0_y25 + 1*z0_y26 + 1*z0_y27 + 1*z0_y28 + 1*z0_y29 + 1*z0_y30 + 1*z0_y31 + 1*z0_y32 + 1*z0_y33 + 1*z0_y34 + 1*z0_y35 + 1*z0_y36 + 1*z0_y37 + 1*z0_y38 + 1*z0_y39 + 1*z0_y40 + 1*z0_y41 = 1:1*z1_y0 + 1*z1_y1 + 1*z1_y2 + 1*z1_y3 + 1*z1_y4 + 1*z1_y5 + 1*z1_y6 + 1*z1_y7 + 1*z1_y8 + 1*z1_y9 + 1*z1_y10 + 1*z1_y11 + 1*z1_y12 + 1*z1_y13 + 1*z1_y14 + 1*z1_y15 + 1*z1_y16 + 1*z1_y17 + 1*z1_y18 + 1*z1_y19 + 1*z1_y20 + 1*z1_y21 + 1*z1_y22 + 1*z1_y23 + 1*z1_y24 + 1*z1_y25 + 1*z1_y26 + 1*z1_y27 + 1*z1_y28 + 1*z1_y29 + 1*z1_y30 + 1*z1_y31 + 1*z1_y32 + 1*z1_y33 + 1*z1_y34 + 1*z1_y35 + 1*z1_y36 + 1*z1_y37 + 1*z1_y38 + 1*z1_y39 + 1*z1_y40 + 1*z1_y41 = 1:1*z2_y0 + 1*z2_y1 + 1*z2_y2 + 1*z2_y3 + 1*z2_y4 + 1*z2_y5 + 1*z2_y6 + 1*z2_y7 + 1*z2_y8 + 1*z2_y9 + 1*z2_y10 + 1*z2_y11 + 1*z2_y12 + 1*z2_y13 + 1*z2_y14 + 1*z2_y15 + 1*z2_y16 + 1*z2_y17 + 1*z2_y18 + 1*z2_y19 + 1*z2_y20 + 1*z2_y21 + 1*z2_y22 + 1*z2_y23 + 1*z2_y24 + 1*z2_y25 + 1*z2_y26 + 1*z2_y27 + 1*z2_y28 + 1*z2_y29 + 1*z2_y30 + 1*z2_y31 + 1*z2_y32 + 1*z2_y33 + 1*z2_y34 + 1*z2_y35 + 1*z2_y36 + 1*z2_y37 + 1*z2_y38 + 1*z2_y39 + 1*z2_y40 + 1*z2_y41 = 1:1*z3_y0 + 1*z3_y1 + 1*z3_y2 + 1*z3_y3 + 1*z3_y4 + 1*z3_y5 + 1*z3_y6 + 1*z3_y7 + 1*z3_y8 + 1*z3_y9 + 1*z3_y10 + 1*z3_y11 + 1*z3_y12 + 1*z3_y13 + 1*z3_y14 + 1*z3_y15 + 1*z3_y16 + 1*z3_y17 + 1*z3_y18 + 1*z3_y19 + 1*z3_y20 + 1*z3_y21 + 1*z3_y22 + 1*z3_y23 + 1*z3_y24 + 1*z3_y25 + 1*z3_y26 + 1*z3_y27 + 1*z3_y28 + 1*z3_y29 + 1*z3_y30 + 1*z3_y31 + 1*z3_y32 + 1*z3_y33 + 1*z3_y34 + 1*z3_y35 + 1*z3_y36 + 1*z3_y37 + 1*z3_y38 + 1*z3_y39 + 1*z3_y40 + 1*z3_y41 = 1:1*z4_y0 + 1*z4_y1 + 1*z4_y2 + 1*z4_y3 + 1*z4_y4 + 1*z4_y5 + 1*z4_y6 + 1*z4_y7 + 1*z4_y8 + 1*z4_y9 + 1*z4_y10 + 1*z4_y11 + 1*z4_y12 + 1*z4_y13 + 1*z4_y14 + 1*z4_y15 + 1*z4_y16 + 1*z4_y17 + 1*z4_y18 + 1*z4_y19 + 1*z4_y20 + 1*z4_y21 + 1*z4_y22 + 1*z4_y23 + 1*z4_y24 + 1*z4_y25 + 1*z4_y26 + 1*z4_y27 + 1*z4_y28 + 1*z4_y29 + 1*z4_y30 + 1*z4_y31 + 1*z4_y32 + 1*z4_y33 + 1*z4_y34 + 1*z4_y35 + 1*z4_y36 + 1*z4_y37 + 1*z4_y38 + 1*z4_y39 + 1*z4_y40 + 1*z4_y41 = 1:1*z5_y0 + 1*z5_y1 + 1*z5_y2 + 1*z5_y3 + 1*z5_y4 + 1*z5_y5 + 1*z5_y6 + 1*z5_y7 + 1*z5_y8 + 1*z5_y9 + 1*z5_y10 + 1*z5_y11 + 1*z5_y12 + 1*z5_y13 + 1*z5_y14 + 1*z5_y15 + 1*z5_y16 + 1*z5_y17 + 1*z5_y18 + 1*z5_y19 + 1*z5_y20 + 1*z5_y21 + 1*z5_y22 + 1*z5_y23 + 1*z5_y24 + 1*z5_y25 + 1*z5_y26 + 1*z5_y27 + 1*z5_y28 + 1*z5_y29 + 1*z5_y30 + 1*z5_y31 + 1*z5_y32 + 1*z5_y33 + 1*z5_y34 + 1*z5_y35 + 1*z5_y36 + 1*z5_y37 + 1*z5_y38 + 1*z5_y39 + 1*z5_y40 + 1*z5_y41 = 1:1*z6_y0 + 1*z6_y1 + 1*z6_y2 + 1*z6_y3 + 1*z6_y4 + 1*z6_y5 + 1*z6_y6 + 1*z6_y7 + 1*z6_y8 + 1*z6_y9 + 1*z6_y10 + 1*z6_y11 + 1*z6_y12 + 1*z6_y13 + 1*z6_y14 + 1*z6_y15 + 1*z6_y16 + 1*z6_y17 + 1*z6_y18 + 1*z6_y19 + 1*z6_y20 + 1*z6_y21 + 1*z6_y22 + 1*z6_y23 + 1*z6_y24 + 1*z6_y25 + 1*z6_y26 + 1*z6_y27 + 1*z6_y28 + 1*z6_y29 + 1*z6_y30 + 1*z6_y31 + 1*z6_y32 + 1*z6_y33 + 1*z6_y34 + 1*z6_y35 + 1*z6_y36 + 1*z6_y37 + 1*z6_y38 + 1*z6_y39 + 1*z6_y40 + 1*z6_y41 = 1:1*z7_y0 + 1*z7_y1 + 1*z7_y2 + 1*z7_y3 + 1*z7_y4 + 1*z7_y5 + 1*z7_y6 + 1*z7_y7 + 1*z7_y8 + 1*z7_y9 + 1*z7_y10 + 1*z7_y11 + 1*z7_y12 + 1*z7_y13 + 1*z7_y14 + 1*z7_y15 + 1*z7_y16 + 1*z7_y17 + 1*z7_y18 + 1*z7_y19 + 1*z7_y20 + 1*z7_y21 + 1*z7_y22 + 1*z7_y23 + 1*z7_y24 + 1*z7_y25 + 1*z7_y26 + 1*z7_y27 + 1*z7_y28 + 1*z7_y29 + 1*z7_y30 + 1*z7_y31 + 1*z7_y32 + 1*z7_y33 + 1*z7_y34 + 1*z7_y35 + 1*z7_y36 + 1*z7_y37 + 1*z7_y38 + 1*z7_y39 + 1*z7_y40 + 1*z7_y41 = 1:1*z0_y0 + -1*y0 <= 0:1*z0_y1 + -1*y1 <= 0:1*z0_y2 + -1*y2 <= 0:1*z0_y3 + -1*y3 <= 0:1*z0_y4 + -1*y4 <= 0:1*z0_y5 + -1*y5 <= 0:1*z0_y6 + -1*y6 <= 0:1*z0_y7 + -1*y7 <= 0:1*z0_y8 + -1*y8 <= 0:1*z0_y9 + -1*y9 <= 0:1*z0_y10 + -1*y10 <= 0:1*z0_y11 + -1*y11 <= 0:1*z0_y12 + -1*y12 <= 0:1*z0_y13 + -1*y13 <= 0:1*z0_y14 + -1*y14 <= 0:1*z0_y15 + -1*y15 <= 0:1*z0_y16 + -1*y16 <= 0:1*z0_y17 + -1*y17 <= 0:1*z0_y18 + -1*y18 <= 0:1*z0_y19 + -1*y19 <= 0:1*z0_y20 + -1*y20 <= 0:1*z0_y21 + -1*y21 <= 0:1*z0_y22 + -1*y22 <= 0:1*z0_y23 + -1*y23 <= 0:1*z0_y24 + -1*y24 <= 0:1*z0_y25 + -1*y25 <= 0:1*z0_y26 + -1*y26 <= 0:1*z0_y27 + -1*y27 <= 0:1*z0_y28 + -1*y28 <= 0:1*z0_y29 + -1*y29 <= 0:1*z0_y30 + -1*y30 <= 0:1*z0_y31 + -1*y31 <= 0:1*z0_y32 + -1*y32 <= 0:1*z0_y33 + -1*y33 <= 0:1*z0_y34 + -1*y34 <= 0:1*z0_y35 + -1*y35 <= 0:1*z0_y36 + -1*y36 <= 0:1*z0_y37 + -1*y37 <= 0:1*z0_y38 + -1*y38 <= 0:1*z0_y39 + -1*y39 <= 0:1*z0_y40 + -1*y40 <= 0:1*z0_y41 + -1*y41 <= 0:1*z1_y0 + -1*y0 <= 0:1*z1_y1 + -1*y1 <= 0:1*z1_y2 + -1*y2 <= 0:1*z1_y3 + -1*y3 <= 0:1*z1_y4 + -1*y4 <= 0:1*z1_y5 + -1*y5 <= 0:1*z1_y6 + -1*y6 <= 0:1*z1_y7 + -1*y7 <= 0:1*z1_y8 + -1*y8 <= 0:1*z1_y9 + -1*y9 <= 0:1*z1_y10 + -1*y10 <= 0:1*z1_y11 + -1*y11 <= 0:1*z1_y12 + -1*y12 <= 0:1*z1_y13 + -1*y13 <= 0:1*z1_y14 + -1*y14 <= 0:1*z1_y15 + -1*y15 <= 0:1*z1_y16 + -1*y16 <= 0:1*z1_y17 + -1*y17 <= 0:1*z1_y18 + -1*y18 <= 0:1*z1_y19 + -1*y19 <= 0:1*z1_y20 + -1*y20 <= 0:1*z1_y21 + -1*y21 <= 0:1*z1_y22 + -1*y22 <= 0:1*z1_y23 + -1*y23 <= 0:1*z1_y24 + -1*y24 <= 0:1*z1_y25 + -1*y25 <= 0:1*z1_y26 + -1*y26 <= 0:1*z1_y27 + -1*y27 <= 0:1*z1_y28 + -1*y28 <= 0:1*z1_y29 + -1*y29 <= 0:1*z1_y30 + -1*y30 <= 0:1*z1_y31 + -1*y31 <= 0:1*z1_y32 + -1*y32 <= 0:1*z1_y33 + -1*y33 <= 0:1*z1_y34 + -1*y34 <= 0:1*z1_y35 + -1*y35 <= 0:1*z1_y36 + -1*y36 <= 0:1*z1_y37 + -1*y37 <= 0:1*z1_y38 + -1*y38 <= 0:1*z1_y39 + -1*y39 <= 0:1*z1_y40 + -1*y40 <= 0:1*z1_y41 + -1*y41 <= 0:1*z2_y0 + -1*y0 <= 0:1*z2_y1 + -1*y1 <= 0:1*z2_y2 + -1*y2 <= 0:1*z2_y3 + -1*y3 <= 0:1*z2_y4 + -1*y4 <= 0:1*z2_y5 + -1*y5 <= 0:1*z2_y6 + -1*y6 <= 0:1*z2_y7 + -1*y7 <= 0:1*z2_y8 + -1*y8 <= 0:1*z2_y9 + -1*y9 <= 0:1*z2_y10 + -1*y10 <= 0:1*z2_y11 + -1*y11 <= 0:1*z2_y12 + -1*y12 <= 0:1*z2_y13 + -1*y13 <= 0:1*z2_y14 + -1*y14 <= 0:1*z2_y15 + -1*y15 <= 0:1*z2_y16 + -1*y16 <= 0:1*z2_y17 + -1*y17 <= 0:1*z2_y18 + -1*y18 <= 0:1*z2_y19 + -1*y19 <= 0:1*z2_y20 + -1*y20 <= 0:1*z2_y21 + -1*y21 <= 0:1*z2_y22 + -1*y22 <= 0:1*z2_y23 + -1*y23 <= 0:1*z2_y24 + -1*y24 <= 0:1*z2_y25 + -1*y25 <= 0:1*z2_y26 + -1*y26 <= 0:1*z2_y27 + -1*y27 <= 0:1*z2_y28 + -1*y28 <= 0:1*z2_y29 + -1*y29 <= 0:1*z2_y30 + -1*y30 <= 0:1*z2_y31 + -1*y31 <= 0:1*z2_y32 + -1*y32 <= 0:1*z2_y33 + -1*y33 <= 0:1*z2_y34 + -1*y34 <= 0:1*z2_y35 + -1*y35 <= 0:1*z2_y36 + -1*y36 <= 0:1*z2_y37 + -1*y37 <= 0:1*z2_y38 + -1*y38 <= 0:1*z2_y39 + -1*y39 <= 0:1*z2_y40 + -1*y40 <= 0:1*z2_y41 + -1*y41 <= 0:1*z3_y0 + -1*y0 <= 0:1*z3_y1 + -1*y1 <= 0:1*z3_y2 + -1*y2 <= 0:1*z3_y3 + -1*y3 <= 0:1*z3_y4 + -1*y4 <= 0:1*z3_y5 + -1*y5 <= 0:1*z3_y6 + -1*y6 <= 0:1*z3_y7 + -1*y7 <= 0:1*z3_y8 + -1*y8 <= 0:1*z3_y9 + -1*y9 <= 0:1*z3_y10 + -1*y10 <= 0:1*z3_y11 + -1*y11 <= 0:1*z3_y12 + -1*y12 <= 0:1*z3_y13 + -1*y13 <= 0:1*z3_y14 + -1*y14 <= 0:1*z3_y15 + -1*y15 <= 0:1*z3_y16 + -1*y16 <= 0:1*z3_y17 + -1*y17 <= 0:1*z3_y18 + -1*y18 <= 0:1*z3_y19 + -1*y19 <= 0:1*z3_y20 + -1*y20 <= 0:1*z3_y21 + -1*y21 <= 0:1*z3_y22 + -1*y22 <= 0:1*z3_y23 + -1*y23 <= 0:1*z3_y24 + -1*y24 <= 0:1*z3_y25 + -1*y25 <= 0:1*z3_y26 + -1*y26 <= 0:1*z3_y27 + -1*y27 <= 0:1*z3_y28 + -1*y28 <= 0:1*z3_y29 + -1*y29 <= 0:1*z3_y30 + -1*y30 <= 0:1*z3_y31 + -1*y31 <= 0:1*z3_y32 + -1*y32 <= 0:1*z3_y33 + -1*y33 <= 0:1*z3_y34 + -1*y34 <= 0:1*z3_y35 + -1*y35 <= 0:1*z3_y36 + -1*y36 <= 0:1*z3_y37 + -1*y37 <= 0:1*z3_y38 + -1*y38 <= 0:1*z3_y39 + -1*y39 <= 0:1*z3_y40 + -1*y40 <= 0:1*z3_y41 + -1*y41 <= 0:1*z4_y0 + -1*y0 <= 0:1*z4_y1 + -1*y1 <= 0:1*z4_y2 + -1*y2 <= 0:1*z4_y3 + -1*y3 <= 0:1*z4_y4 + -1*y4 <= 0:1*z4_y5 + -1*y5 <= 0:1*z4_y6 + -1*y6 <= 0:1*z4_y7 + -1*y7 <= 0:1*z4_y8 + -1*y8 <= 0:1*z4_y9 + -1*y9 <= 0:1*z4_y10 + -1*y10 <= 0:1*z4_y11 + -1*y11 <= 0:1*z4_y12 + -1*y12 <= 0:1*z4_y13 + -1*y13 <= 0:1*z4_y14 + -1*y14 <= 0:1*z4_y15 + -1*y15 <= 0:1*z4_y16 + -1*y16 <= 0:1*z4_y17 + -1*y17 <= 0:1*z4_y18 + -1*y18 <= 0:1*z4_y19 + -1*y19 <= 0:1*z4_y20 + -1*y20 <= 0:1*z4_y21 + -1*y21 <= 0:1*z4_y22 + -1*y22 <= 0:1*z4_y23 + -1*y23 <= 0:1*z4_y24 + -1*y24 <= 0:1*z4_y25 + -1*y25 <= 0:1*z4_y26 + -1*y26 <= 0:1*z4_y27 + -1*y27 <= 0:1*z4_y28 + -1*y28 <= 0:1*z4_y29 + -1*y29 <= 0:1*z4_y30 + -1*y30 <= 0:1*z4_y31 + -1*y31 <= 0:1*z4_y32 + -1*y32 <= 0:1*z4_y33 + -1*y33 <= 0:1*z4_y34 + -1*y34 <= 0:1*z4_y35 + -1*y35 <= 0:1*z4_y36 + -1*y36 <= 0:1*z4_y37 + -1*y37 <= 0:1*z4_y38 + -1*y38 <= 0:1*z4_y39 + -1*y39 <= 0:1*z4_y40 + -1*y40 <= 0:1*z4_y41 + -1*y41 <= 0:1*z5_y0 + -1*y0 <= 0:1*z5_y1 + -1*y1 <= 0:1*z5_y2 + -1*y2 <= 0:1*z5_y3 + -1*y3 <= 0:1*z5_y4 + -1*y4 <= 0:1*z5_y5 + -1*y5 <= 0:1*z5_y6 + -1*y6 <= 0:1*z5_y7 + -1*y7 <= 0:1*z5_y8 + -1*y8 <= 0:1*z5_y9 + -1*y9 <= 0:1*z5_y10 + -1*y10 <= 0:1*z5_y11 + -1*y11 <= 0:1*z5_y12 + -1*y12 <= 0:1*z5_y13 + -1*y13 <= 0:1*z5_y14 + -1*y14 <= 0:1*z5_y15 + -1*y15 <= 0:1*z5_y16 + -1*y16 <= 0:1*z5_y17 + -1*y17 <= 0:1*z5_y18 + -1*y18 <= 0:1*z5_y19 + -1*y19 <= 0:1*z5_y20 + -1*y20 <= 0:1*z5_y21 + -1*y21 <= 0:1*z5_y22 + -1*y22 <= 0:1*z5_y23 + -1*y23 <= 0:1*z5_y24 + -1*y24 <= 0:1*z5_y25 + -1*y25 <= 0:1*z5_y26 + -1*y26 <= 0:1*z5_y27 + -1*y27 <= 0:1*z5_y28 + -1*y28 <= 0:1*z5_y29 + -1*y29 <= 0:1*z5_y30 + -1*y30 <= 0:1*z5_y31 + -1*y31 <= 0:1*z5_y32 + -1*y32 <= 0:1*z5_y33 + -1*y33 <= 0:1*z5_y34 + -1*y34 <= 0:1*z5_y35 + -1*y35 <= 0:1*z5_y36 + -1*y36 <= 0:1*z5_y37 + -1*y37 <= 0:1*z5_y38 + -1*y38 <= 0:1*z5_y39 + -1*y39 <= 0:1*z5_y40 + -1*y40 <= 0:1*z5_y41 + -1*y41 <= 0:1*z6_y0 + -1*y0 <= 0:1*z6_y1 + -1*y1 <= 0:1*z6_y2 + -1*y2 <= 0:1*z6_y3 + -1*y3 <= 0:1*z6_y4 + -1*y4 <= 0:1*z6_y5 + -1*y5 <= 0:1*z6_y6 + -1*y6 <= 0:1*z6_y7 + -1*y7 <= 0:1*z6_y8 + -1*y8 <= 0:1*z6_y9 + -1*y9 <= 0:1*z6_y10 + -1*y10 <= 0:1*z6_y11 + -1*y11 <= 0:1*z6_y12 + -1*y12 <= 0:1*z6_y13 + -1*y13 <= 0:1*z6_y14 + -1*y14 <= 0:1*z6_y15 + -1*y15 <= 0:1*z6_y16 + -1*y16 <= 0:1*z6_y17 + -1*y17 <= 0:1*z6_y18 + -1*y18 <= 0:1*z6_y19 + -1*y19 <= 0:1*z6_y20 + -1*y20 <= 0:1*z6_y21 + -1*y21 <= 0:1*z6_y22 + -1*y22 <= 0:1*z6_y23 + -1*y23 <= 0:1*z6_y24 + -1*y24 <= 0:1*z6_y25 + -1*y25 <= 0:1*z6_y26 + -1*y26 <= 0:1*z6_y27 + -1*y27 <= 0:1*z6_y28 + -1*y28 <= 0:1*z6_y29 + -1*y29 <= 0:1*z6_y30 + -1*y30 <= 0:1*z6_y31 + -1*y31 <= 0:1*z6_y32 + -1*y32 <= 0:1*z6_y33 + -1*y33 <= 0:1*z6_y34 + -1*y34 <= 0:1*z6_y35 + -1*y35 <= 0:1*z6_y36 + -1*y36 <= 0:1*z6_y37 + -1*y37 <= 0:1*z6_y38 + -1*y38 <= 0:1*z6_y39 + -1*y39 <= 0:1*z6_y40 + -1*y40 <= 0:1*z6_y41 + -1*y41 <= 0:1*z7_y0 + -1*y0 <= 0:1*z7_y1 + -1*y1 <= 0:1*z7_y2 + -1*y2 <= 0:1*z7_y3 + -1*y3 <= 0:1*z7_y4 + -1*y4 <= 0:1*z7_y5 + -1*y5 <= 0:1*z7_y6 + -1*y6 <= 0:1*z7_y7 + -1*y7 <= 0:1*z7_y8 + -1*y8 <= 0:1*z7_y9 + -1*y9 <= 0:1*z7_y10 + -1*y10 <= 0:1*z7_y11 + -1*y11 <= 0:1*z7_y12 + -1*y12 <= 0:1*z7_y13 + -1*y13 <= 0:1*z7_y14 + -1*y14 <= 0:1*z7_y15 + -1*y15 <= 0:1*z7_y16 + -1*y16 <= 0:1*z7_y17 + -1*y17 <= 0:1*z7_y18 + -1*y18 <= 0:1*z7_y19 + -1*y19 <= 0:1*z7_y20 + -1*y20 <= 0:1*z7_y21 + -1*y21 <= 0:1*z7_y22 + -1*y22 <= 0:1*z7_y23 + -1*y23 <= 0:1*z7_y24 + -1*y24 <= 0:1*z7_y25 + -1*y25 <= 0:1*z7_y26 + -1*y26 <= 0:1*z7_y27 + -1*y27 <= 0:1*z7_y28 + -1*y28 <= 0:1*z7_y29 + -1*y29 <= 0:1*z7_y30 + -1*y30 <= 0:1*z7_y31 + -1*y31 <= 0:1*z7_y32 + -1*y32 <= 0:1*z7_y33 + -1*y33 <= 0:1*z7_y34 + -1*y34 <= 0:1*z7_y35 + -1*y35 <= 0:1*z7_y36 + -1*y36 <= 0:1*z7_y37 + -1*y37 <= 0:1*z7_y38 + -1*y38 <= 0:1*z7_y39 + -1*y39 <= 0:1*z7_y40 + -1*y40 <= 0:1*z7_y41 + -1*y41 <= 0:1*z0_y0 + 1*z1_y0 + 1*z2_y0 + 1*z3_y0 + 1*z4_y0 + 1*z5_y0 + 1*z6_y0 + 1*z7_y0 + -1*y0 >= 0:1*z0_y1 + 1*z1_y1 + 1*z2_y1 + 1*z3_y1 + 1*z4_y1 + 1*z5_y1 + 1*z6_y1 + 1*z7_y1 + -1*y1 >= 0:1*z0_y2 + 1*z1_y2 + 1*z2_y2 + 1*z3_y2 + 1*z4_y2 + 1*z5_y2 + 1*z6_y2 + 1*z7_y2 + -1*y2 >= 0:1*z0_y3 + 1*z1_y3 + 1*z2_y3 + 1*z3_y3 + 1*z4_y3 + 1*z5_y3 + 1*z6_y3 + 1*z7_y3 + -1*y3 >= 0:1*z0_y4 + 1*z1_y4 + 1*z2_y4 + 1*z3_y4 + 1*z4_y4 + 1*z5_y4 + 1*z6_y4 + 1*z7_y4 + -1*y4 >= 0:1*z0_y5 + 1*z1_y5 + 1*z2_y5 + 1*z3_y5 + 1*z4_y5 + 1*z5_y5 + 1*z6_y5 + 1*z7_y5 + -1*y5 >= 0:1*z0_y6 + 1*z1_y6 + 1*z2_y6 + 1*z3_y6 + 1*z4_y6 + 1*z5_y6 + 1*z6_y6 + 1*z7_y6 + -1*y6 >= 0:1*z0_y7 + 1*z1_y7 + 1*z2_y7 + 1*z3_y7 + 1*z4_y7 + 1*z5_y7 + 1*z6_y7 + 1*z7_y7 + -1*y7 >= 0:1*z0_y8 + 1*z1_y8 + 1*z2_y8 + 1*z3_y8 + 1*z4_y8 + 1*z5_y8 + 1*z6_y8 + 1*z7_y8 + -1*y8 >= 0:1*z0_y9 + 1*z1_y9 + 1*z2_y9 + 1*z3_y9 + 1*z4_y9 + 1*z5_y9 + 1*z6_y9 + 1*z7_y9 + -1*y9 >= 0:1*z0_y10 + 1*z1_y10 + 1*z2_y10 + 1*z3_y10 + 1*z4_y10 + 1*z5_y10 + 1*z6_y10 + 1*z7_y10 + -1*y10 >= 0:1*z0_y11 + 1*z1_y11 + 1*z2_y11 + 1*z3_y11 + 1*z4_y11 + 1*z5_y11 + 1*z6_y11 + 1*z7_y11 + -1*y11 >= 0:1*z0_y12 + 1*z1_y12 + 1*z2_y12 + 1*z3_y12 + 1*z4_y12 + 1*z5_y12 + 1*z6_y12 + 1*z7_y12 + -1*y12 >= 0:1*z0_y13 + 1*z1_y13 + 1*z2_y13 + 1*z3_y13 + 1*z4_y13 + 1*z5_y13 + 1*z6_y13 + 1*z7_y13 + -1*y13 >= 0:1*z0_y14 + 1*z1_y14 + 1*z2_y14 + 1*z3_y14 + 1*z4_y14 + 1*z5_y14 + 1*z6_y14 + 1*z7_y14 + -1*y14 >= 0:1*z0_y15 + 1*z1_y15 + 1*z2_y15 + 1*z3_y15 + 1*z4_y15 + 1*z5_y15 + 1*z6_y15 + 1*z7_y15 + -1*y15 >= 0:1*z0_y16 + 1*z1_y16 + 1*z2_y16 + 1*z3_y16 + 1*z4_y16 + 1*z5_y16 + 1*z6_y16 + 1*z7_y16 + -1*y16 >= 0:1*z0_y17 + 1*z1_y17 + 1*z2_y17 + 1*z3_y17 + 1*z4_y17 + 1*z5_y17 + 1*z6_y17 + 1*z7_y17 + -1*y17 >= 0:1*z0_y18 + 1*z1_y18 + 1*z2_y18 + 1*z3_y18 + 1*z4_y18 + 1*z5_y18 + 1*z6_y18 + 1*z7_y18 + -1*y18 >= 0:1*z0_y19 + 1*z1_y19 + 1*z2_y19 + 1*z3_y19 + 1*z4_y19 + 1*z5_y19 + 1*z6_y19 + 1*z7_y19 + -1*y19 >= 0:1*z0_y20 + 1*z1_y20 + 1*z2_y20 + 1*z3_y20 + 1*z4_y20 + 1*z5_y20 + 1*z6_y20 + 1*z7_y20 + -1*y20 >= 0:1*z0_y21 + 1*z1_y21 + 1*z2_y21 + 1*z3_y21 + 1*z4_y21 + 1*z5_y21 + 1*z6_y21 + 1*z7_y21 + -1*y21 >= 0:1*z0_y22 + 1*z1_y22 + 1*z2_y22 + 1*z3_y22 + 1*z4_y22 + 1*z5_y22 + 1*z6_y22 + 1*z7_y22 + -1*y22 >= 0:1*z0_y23 + 1*z1_y23 + 1*z2_y23 + 1*z3_y23 + 1*z4_y23 + 1*z5_y23 + 1*z6_y23 + 1*z7_y23 + -1*y23 >= 0:1*z0_y24 + 1*z1_y24 + 1*z2_y24 + 1*z3_y24 + 1*z4_y24 + 1*z5_y24 + 1*z6_y24 + 1*z7_y24 + -1*y24 >= 0:1*z0_y25 + 1*z1_y25 + 1*z2_y25 + 1*z3_y25 + 1*z4_y25 + 1*z5_y25 + 1*z6_y25 + 1*z7_y25 + -1*y25 >= 0:1*z0_y26 + 1*z1_y26 + 1*z2_y26 + 1*z3_y26 + 1*z4_y26 + 1*z5_y26 + 1*z6_y26 + 1*z7_y26 + -1*y26 >= 0:1*z0_y27 + 1*z1_y27 + 1*z2_y27 + 1*z3_y27 + 1*z4_y27 + 1*z5_y27 + 1*z6_y27 + 1*z7_y27 + -1*y27 >= 0:1*z0_y28 + 1*z1_y28 + 1*z2_y28 + 1*z3_y28 + 1*z4_y28 + 1*z5_y28 + 1*z6_y28 + 1*z7_y28 + -1*y28 >= 0:1*z0_y29 + 1*z1_y29 + 1*z2_y29 + 1*z3_y29 + 1*z4_y29 + 1*z5_y29 + 1*z6_y29 + 1*z7_y29 + -1*y29 >= 0:1*z0_y30 + 1*z1_y30 + 1*z2_y30 + 1*z3_y30 + 1*z4_y30 + 1*z5_y30 + 1*z6_y30 + 1*z7_y30 + -1*y30 >= 0:1*z0_y31 + 1*z1_y31 + 1*z2_y31 + 1*z3_y31 + 1*z4_y31 + 1*z5_y31 + 1*z6_y31 + 1*z7_y31 + -1*y31 >= 0:1*z0_y32 + 1*z1_y32 + 1*z2_y32 + 1*z3_y32 + 1*z4_y32 + 1*z5_y32 + 1*z6_y32 + 1*z7_y32 + -1*y32 >= 0:1*z0_y33 + 1*z1_y33 + 1*z2_y33 + 1*z3_y33 + 1*z4_y33 + 1*z5_y33 + 1*z6_y33 + 1*z7_y33 + -1*y33 >= 0:1*z0_y34 + 1*z1_y34 + 1*z2_y34 + 1*z3_y34 + 1*z4_y34 + 1*z5_y34 + 1*z6_y34 + 1*z7_y34 + -1*y34 >= 0:1*z0_y35 + 1*z1_y35 + 1*z2_y35 + 1*z3_y35 + 1*z4_y35 + 1*z5_y35 + 1*z6_y35 + 1*z7_y35 + -1*y35 >= 0:1*z0_y36 + 1*z1_y36 + 1*z2_y36 + 1*z3_y36 + 1*z4_y36 + 1*z5_y36 + 1*z6_y36 + 1*z7_y36 + -1*y36 >= 0:1*z0_y37 + 1*z1_y37 + 1*z2_y37 + 1*z3_y37 + 1*z4_y37 + 1*z5_y37 + 1*z6_y37 + 1*z7_y37 + -1*y37 >= 0:1*z0_y38 + 1*z1_y38 + 1*z2_y38 + 1*z3_y38 + 1*z4_y38 + 1*z5_y38 + 1*z6_y38 + 1*z7_y38 + -1*y38 >= 0:1*z0_y39 + 1*z1_y39 + 1*z2_y39 + 1*z3_y39 + 1*z4_y39 + 1*z5_y39 + 1*z6_y39 + 1*z7_y39 + -1*y39 >= 0:1*z0_y40 + 1*z1_y40 + 1*z2_y40 + 1*z3_y40 + 1*z4_y40 + 1*z5_y40 + 1*z6_y40 + 1*z7_y40 + -1*y40 >= 0:1*z0_y41 + 1*z1_y41 + 1*z2_y41 + 1*z3_y41 + 1*z4_y41 + 1*z5_y41 + 1*z6_y41 + 1*z7_y41 + -1*y41 >= 0";
    
    public static void bigProblem(){
    	SolverFactory factory = new SolverFactoryLpSolve(); // use lp_solve
		Map<Object, Object> params = factory.getParameters();
		//System.out.println("PARAMS : " + params);
		factory.setParameter(Solver.VERBOSE, 1);
		factory.setParameter(Solver.TIMEOUT, 100); // set timeout to 100 seconds
		//System.out.println("PARAMS after: " + params);
	
		String [] terms = objectiveString.split("\\ \\+\\ ");
		Problem problem = new Problem();
		
		Linear linear = new Linear();
		for(String term : terms){
			String [] sp = term.split("\\*");
			double coeff = Double.parseDouble(sp[0]);
			linear.add(coeff, sp[1].trim());
			//System.out.println(coeff + " -- " + sp[1]);
		}
		
		problem.setObjective(linear, OptType.MAX);
		
		String [] constraints = constraintsString.split("\\:");
		for(String constraint : constraints){
			//System.out.println(constraint);
			linear = new Linear();
			String splitRegex = null;
			String operator = "";
			if(constraint.contains(" <= ")){
				splitRegex = "\\<\\=";
				operator = "<=";
			}
			else if(constraint.contains(" = ")){
				splitRegex = "\\=";
				operator = "=";
			}
			else if(constraint.contains(" >= ")){
				splitRegex = "\\>\\=";
				operator = ">=";
			}
			String sp[] = constraint.split(splitRegex);
			String termss[] = sp[0].split("\\ \\+\\ ");
			linear = new Linear();
			for(String t : termss){
				String [] spp = t.split("\\*");
				int coeff = Integer.parseInt(spp[0]);
				linear.add(coeff, spp[1].trim());
				//System.out.println(t);
			}
			problem.add(linear, operator, Integer.parseInt(sp[1].trim()));		
		}
		
//		System.out.println("No. of constraints : " + constraints.length);
//		System.out.println("No. of constraints : " + problem.getConstraintsCount());
		
		for(Object var : problem.getVariables())
			problem.setVarType(var, Boolean.class);
		
		Solver solver = factory.get(); // you should use this solver only once for one problem
		Result result = solver.solve(problem);

		System.out.println("----------------------------------------------");
		
		System.out.println(result);
		
		System.out.println("-----------------------------------------------");
		
//		System.out.println("Num of variables : " + problem.getVariablesCount());
//		System.out.println("Num of Constraints : " + problem.getConstraintsCount());
//		System.out.println("Objective Function : ");
//		System.out.println(problem.getObjective());
//		System.out.println("Constraints : ");
//		for(Constraint c : problem.getConstraints())
//			System.out.println(c);
		
    	System.exit(0);
    }
    
	public static void main(String args[]){
	    //logger.info("java.library.path : " + System.getProperty("java.library.path"));
	    //logger.info("java.class.path : " + System.getProperty("java.class.path"));
		
		bigProblem();
		
		SolverFactory factory = new SolverFactoryLpSolve(); // use lp_solve
		Map<Object, Object> params = factory.getParameters();
		System.out.println("PARAMS : " + params);
		factory.setParameter(Solver.VERBOSE, 0);
		factory.setParameter(Solver.TIMEOUT, 100); // set timeout to 100 seconds
		System.out.println("PARAMS after: " + params);
		
		
		/**
		* Constructing a Problem: 
		* Maximize: 143x+60y 
		* Subject to: 
		* 120x+210y <= 15000 
		* 110x+30y <= 4000 
		* x+y <= 75
		* 
		* With x,y being integers
		* 
		*/
		Problem problem = new Problem();

		Linear linear = new Linear();
		linear.add(143, "x");
		linear.add(60, "y");

		problem.setObjective(linear, OptType.MAX);

		linear = new Linear();
		linear.add(120, "x");
		linear.add(210, "y");

		problem.add(linear, "<=", 15000);

		linear = new Linear();
		linear.add(110, "x");
		linear.add(30, "y");

		problem.add(linear, "<=", 4000);

		linear = new Linear();
		linear.add(1, "x");
		linear.add(1, "y");

		problem.add(linear, "<=", 75);

		problem.setVarType("x", Integer.class);
		problem.setVarType("y", Integer.class);

		Solver solver = factory.get(); // you should use this solver only once for one problem
		Result result = solver.solve(problem);

		System.out.println(result);

		/**
		* Extend the problem with x <= 16 and solve it again
		*/
		problem.setVarUpperBound("x", 16);

		solver = factory.get();
		result = solver.solve(problem);

		System.out.println(result);
	}
}
