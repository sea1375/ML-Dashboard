import time
import numpy as np
import pandas as pd
import holoviews as hv
from holoviews import dim, opts


# import streamz
# import streamz.dataframe

# from holoviews.streams import Pipe, Buffer

"""bokeh serve --allow-websocket-origin='*' --port 5006 test_streaming.py"""

renderer = hv.renderer('bokeh')


# epoch_loss = pd.DataFrame({'epoch': [], 'loss': []}, columns=['epoch', 'loss'])
# epoch_micro = pd.DataFrame({'epoch': [], 'f1_micro': []}, columns=['epoch', 'f1_micro'])
# epoch_macro = pd.DataFrame({'epoch': [], 'f1_macro': []}, columns=['epoch', 'f1_macro'])
# epoch_time = pd.DataFrame({'epoch': [], 'time': []}, columns=['epoch', 'time'])

# pipe_loss = Pipe(data=epochs_loss)
# pipe_micro = Pipe(data=epochs_micro)
# pipe_macro = Pipe(data=epochs_macro)
# pipe_time = Pipe(data=epochs_time)



def get_loss_data():
	d = {"Epochs":i,"Loss":losses[i]}
	df=pd.DataFrame(d.items(), columns=['Epochs','Loss'])

	return df
	
def get_time_data():
	d = {"Epochs":i,"Time":losses[i]}
	df=pd.DataFrame(d.items(), columns=['Epochs','Time'])
	return df

def get_micro_data():
	d = {"Epochs":i,"Score":losses[i]}
	df=pd.DataFrame(d.items(), columns=['Epochs','Score'])
	return df

def get_macro_data():
	d = {"Epochs":i,"Score":losses[i]}
	df=pd.DataFrame(d.items(),  columns=['Epochs','Score'])
	return df
	


def macro_box(data):
    return hv.Curve(data, 'Epochs', 'Score', label='Macro F1 Score')

def micro_box(data):
    return hv.Curve(data, 'Epochs', 'Score', label='Micro F1 Score')

def time_box(data):
    return hv.Curve(data, 'Epochs', 'Time', label='Time Estimation')

def loss_box(data):
    return hv.Curve(data, 'Epochs', 'Loss', label='Convergence')


def cb_loss_time():
    loss_stream.send(get_loss_data())
    time_stream.send(get_time_data())

def cb_micro_macro():
    micro_stream.send(get_micro_data())
    macro_stream.send(get_macro_data())

# Define DynamicMaps and display plot

i=0
losses=[1.3275,1.11855,1.04693,0.99589,0.96186,0.93409,0.90698,0.89055,0.87493,0.8652,0.84688,0.83957,0.82635,0.82291,0.80957,0.80137,0.79266,0.78657,0.78969,0.77722,0.77541,0.77073,0.76204,0.75995,0.74467,0.76001,0.74359,0.7387,0.74616,0.73368,0.73932,0.735,0.73062,0.72712,0.72433,0.72457,0.72445,0.71542,0.71151,0.71813,0.70696,0.70706,0.70425,0.70221,0.71081,0.69971,0.69754,0.69267,0.69544,0.6868,0.68685,0.69339,0.6891,0.70096,0.69346,0.69201,0.69263,0.67551,0.67638,0.67525,0.68387,0.67593,0.67239,0.6755,0.67392,0.66521,0.67923,0.67486,0.6711,0.66967,0.65978,0.6681,0.67572,0.65592,0.65573,0.65146,0.66311,0.65875,0.66158,0.65786,0.65221,0.65115,0.65613,0.65625,0.65474,0.6472,0.63679,0.6335,0.63884,0.66326,0.64423,0.64284,0.64054,0.64087,0.64333,0.63727,0.64504,0.63774,0.63348,0.63561,0.64028,0.63349,0.62979,0.63086,0.62785,0.62395,0.6239,0.61621,0.6203,0.61776,0.62908,0.63278,0.62858,0.6211,0.61874,0.6226,0.62045,0.61101,0.60776,0.62459,0.62525,0.61969,0.62812,0.60653,0.60601,0.60973,0.60215,0.61191,0.61143,0.61717,0.62279,0.60762,0.62425,0.61668,0.61094,0.61133,0.61468,0.60715,0.60296,0.59648,0.59902,0.60245,0.60647,0.60295,0.59491,0.59056,0.5959,0.58997,0.59596,0.59694,0.59979,0.60334,0.60099,0.59324,0.60785,0.59666,0.59321,0.5931,0.60239,0.59463,0.59259,0.60809,0.60598,0.58622,0.59034,0.57938,0.58402,0.59172,0.58373,0.58308,0.58481,0.58279,0.59357,0.58793,0.58059,0.57482,0.58367,0.58756,0.57897,0.58554,0.58307,0.5795,0.57181,0.57018,0.58485,0.59803,0.57607,0.5692,0.56803,0.57557,0.57097,0.57482,0.57739,0.57291,0.57167,0.56728,0.56946,0.57302,0.57398,0.57374,0.57137,0.57364,0.56895,0.5621,0.56723,0.57197,0.57868,0.57767,0.57618,0.56338,0.56245,0.5595,0.56665,0.56009,0.56071,0.56781,0.56756,0.56817,0.56766,0.55935,0.56686,0.57086,0.57923,0.56807,0.56187,0.55842,0.55243,0.5629,0.57095,0.587,0.572,0.55858,0.56433,0.56516,0.55224,0.54346,0.55418,0.55862,0.55694,0.56249,0.5641,0.56159,0.56309,0.55348,0.54794,0.55503,0.5575,0.55508,0.55617,0.54834,0.54834,0.54407,0.54371,0.54642,0.55609,0.57458,0.57764,0.55801,0.5455,0.54231,0.54619,0.55575,0.55376,0.56072,0.55001,0.54389,0.54001,0.53917,0.54226,0.53718,0.54569,0.54747,0.55043,0.55274,0.54787,0.5478,0.54231,0.5415,0.54225,0.5428,0.54042,0.54019,0.5485,0.54422,0.54529,0.54055,0.5363,0.54017,0.54048,0.53499,0.52945,0.55487,0.54496,0.53941,0.52748,0.55446,0.53933,0.53885,0.5387,0.53785,0.5332,0.5346,0.54024,0.53134,0.53203,0.52829,0.5279,0.53277,0.53568,0.5406,0.52938,0.53872,0.53729,0.53807,0.52859,0.52662,0.5298,0.53086,0.52705,0.5214,0.52438,0.53408,0.52621,0.52914,0.52517,0.53441,0.525,0.5356,0.53559,0.53206,0.52406,0.54837,0.53529,0.52547,0.52611,0.53716,0.53524,0.52531,0.52106,0.51848,0.52231,0.52041,0.52927,0.51974,0.52503,0.52788,0.52737,0.52442,0.51954,0.52063,0.51974,0.51609,0.51817,0.54218,0.5309,0.52554,0.53701,0.52884,0.53694,0.53531,0.52651,0.52584,0.52035,0.51516,0.53131,0.52945,0.52109,0.52129,0.52148,0.52078,0.52164,0.52203,0.52497,0.52386,0.53438,0.53661,0.52493,0.53115,0.52068,0.51878,0.52263,0.51428,0.51786,0.51635,0.51905,0.51542,0.51266,0.51442,0.52444,0.52284,0.51427,0.50849,0.51546,0.53006,0.51887,0.51513,0.52976,0.52038,0.51519,0.51421,0.50804,0.51524,0.51758,0.51909,0.52708,0.52102,0.51725,0.5451,0.52789,0.52117,0.52254,0.51694,0.51768,0.53934,0.52103,0.52553,0.51687,0.52846,0.51959,0.51711,0.51404,0.51824,0.52136,0.52342,0.51902,0.52492,0.5252,0.52078,0.51479,0.51686,0.51555,0.52299,0.53186,0.52237,0.51279,0.51489,0.52477,0.52355,0.52233,0.52744,0.51737,0.51751,0.51311,0.51373,0.51415,0.52153,0.52662,0.51295,0.50901,0.51481,0.51923,0.51694,0.51505,0.51568,0.51481,0.51796,0.52692,0.52205,0.52076,0.51591,0.51102,0.50961,0.50886,0.51062,0.51392,0.51635,0.51053,0.50685,0.51362,0.50761,0.50182,0.51313,0.50745,0.50912,0.50623,0.51565,0.5139,0.50456,0.51249,0.51156,0.50869,0.5067,0.50927,0.50466,0.5032,0.51623,0.52475,0.51445,0.51579,0.51729,0.50523,0.52683,0.52853,0.51974,0.51257,0.51835,0.51223,0.51806,0.5126,0.52064,0.52351,0.52319,0.5137,0.51748,0.51423,0.50669,0.50186,0.49802,0.51024,0.50466,0.51042,0.51708,0.51258,0.50597,0.49709,0.49935,0.49991,0.50831,0.51116,0.50672,0.50385,0.50893,0.51684,0.50864,0.50865,0.50724,0.51098,0.50692,0.50279,0.5123,0.51076,0.51695,0.5132,0.53162,0.5307,0.52108,0.5145,0.5089,0.50034,0.50021,0.50169,0.4963,0.49798,0.49782,0.49475,0.50319,0.50378,0.51354,0.51129,0.51695,0.50892,0.51323,0.50989,0.50711,0.50579,0.50384,0.50857,0.50773,0.5101,0.51502,0.51315,0.50752,0.51297,0.50275,0.5051,0.51102,0.50616,0.51184,0.52098,0.51488,0.50082,0.50517,0.50202,0.49818,0.50131,0.5171,0.51553,0.50158,0.50605,0.508,0.50157,0.4995,0.4936,0.49508,0.50288,0.50817,0.51124,0.51714,0.5006,0.50171,0.50347,0.50239,0.49558,0.50042,0.49637,0.49559,0.49827,0.50855,0.50575,0.50983,0.50228,0.49939,0.50327,0.49771,0.49992,0.4896,0.49649,0.50286,0.50053,0.50186,0.49466,0.48996,0.49588,0.50122,0.50395,0.49737,0.49848,0.50504,0.49965,0.51553,0.50949,0.49746,0.49376,0.49338,0.49039,0.49233,0.49155,0.49435,0.5008,0.49547,0.49421,0.49628,0.49072,0.50615,0.4983,0.50085,0.49919,0.49407,0.49249,0.49183,0.48762,0.49304,0.49195,0.49496,0.49449,0.49677,0.49229,0.49241,0.49513,0.49947,0.4902,0.48882,0.49358,0.50207,0.48917,0.49783,0.4923,0.48999,0.49409,0.49165,0.48488,0.49179,0.48964,0.49686,0.49844,0.51008,0.50146,0.51133,0.50037,0.50232,0.49603,0.50942,0.50817,0.49973,0.50757,0.49568,0.49159,0.49597,0.4963,0.48782,0.49245,0.49258,0.49656,0.4967,0.50309,0.508,0.51406,0.51263,0.50732,0.50931,0.50407,0.50597,0.49288,0.48671,0.48607,0.4878,0.49033,0.49647,0.50154,0.50591,0.49548,0.49572,0.49526,0.50149,0.4945,0.50051,0.49931,0.49214,0.49072,0.49227,0.49523,0.48921,0.49464,0.49302,0.49657,0.49137,0.49543,0.50507,0.50295,0.49513,0.49478,0.49364,0.50005,0.49456,0.49348,0.49638,0.50297,0.501,0.50323,0.50488,0.49585,0.49522,0.50085,0.49953,0.49274,0.48829,0.48794,0.48449,0.49486,0.49259,0.49119,0.48897,0.4911,0.51127,0.51361,0.50939,0.49911,0.49541,0.49241,0.4918,0.49055,0.49332,0.4906,0.48818,0.49105,0.49781,0.48894,0.49594,0.49007,0.48682,0.494,0.49465,0.49239,0.49543,0.48834,0.49676,0.49749,0.49314,0.5029,0.49293,0.49054,0.49485,0.49274,0.49239,0.49508,0.49513,0.51097,0.51561,0.50155,0.49682,0.48871,0.48548,0.48675,0.49071,0.50516,0.49401,0.49272,0.48644,0.49894,0.49272,0.49205,0.48836,0.49349,0.49659,0.49397,0.4879,0.49046,0.48786,0.48872,0.48457,0.48367,0.49142,0.48907,0.4845,0.48639,0.48698,0.49473,0.49806,0.4895,0.48856,0.49119,0.50097,0.51089,0.50253,0.49422,0.4889,0.48347,0.49065,0.48604,0.49028,0.49306,0.49767,0.49016,0.495,0.50199,0.51489,0.49918,0.50858,0.49752,0.48716,0.48364,0.49135,0.49275,0.48535,0.48504,0.48592,0.48323,0.48485,0.4843,0.48735,0.48343,0.49586,0.49684,0.49287,0.49669,0.49314,0.49358,0.48418,0.48812,0.48722,0.50596,0.49626,0.48754,0.48638,0.48307,0.48736,0.50611,0.49894,0.49641,0.49151,0.49765,0.49045,0.49375,0.48792,0.48496,0.4883,0.4847,0.50114,0.49012,0.49458,0.49043,0.48731,0.4859,0.48661,0.48435,0.47943,0.49611,0.48439,0.48053,0.48162,0.48277,0.48085,0.48061,0.48271,0.48098,0.48836,0.49468,0.49831,0.49763,0.49306,0.48818,0.4826,0.48955,0.49209,0.48633,0.49007,0.4939,0.49269,0.48627,0.48533,0.48991,0.48582,0.48662,0.48618,0.48231,0.48652,0.48369,0.48266,0.48823,0.48365,0.48486,0.49133,0.48346,0.48262,0.48691,0.48632,0.4822,0.4826,0.48093,0.476,0.48406,0.48252,0.48422,0.47967,0.48321,0.48572,0.48886,0.48836,0.4866,0.48554,0.48067,0.47999,0.47868,0.48147,0.47785,0.48381,0.48788,0.4917,0.48423,0.48569,0.50091,0.48993,0.48935,0.50267,0.50021,0.48308,0.48103,0.4792,0.48752,0.48873,0.48159,0.47934,0.4807,0.47677,0.48106,0.48091,0.4872,0.48778,0.48319,0.48953,0.48372,0.49675,0.49229,0.48511,0.48103,0.47427,0.47891,0.48236,0.4806,0.48543,0.49911,0.48768,0.48832,0.4935,0.48878,0.4942,0.49846,0.49251,0.49333,0.48899,0.48654,0.48747,0.4862,0.48934,0.494,0.48603,0.48329,0.47957,0.47973,0.48418,0.48517,0.48405,0.47781,0.48069,0.48304,0.48013,0.479,0.4806,0.49227,0.48609]
micros=[0.44995,0.52131,0.61732,0.68136,0.71198,0.73941,0.75916,0.76441,0.78748,0.7887,0.81517,0.8149,0.8164,0.82036,0.82798,0.82948,0.84338,0.84807,0.85285,0.84681,0.84302,0.85493,0.85852,0.86404,0.86881,0.86109,0.86673,0.88136,0.88042,0.88244,0.87776,0.8797,0.87835,0.87692,0.88427,0.88102,0.88191,0.89395,0.88101,0.88523,0.88386,0.89451,0.89459,0.8818,0.88351,0.90319,0.89422,0.89653,0.89527,0.90035,0.90188,0.90664,0.89958,0.90175,0.90543,0.89465,0.90273,0.91134,0.90903,0.90733,0.90783,0.90232,0.91152,0.90282,0.90935,0.91315,0.90661,0.91263,0.91918,0.9094,0.91296,0.91119,0.91269,0.92128,0.92663,0.91631,0.92692,0.92417,0.92105,0.92102,0.92799,0.91839,0.92307,0.92669,0.93145,0.93231,0.93502,0.92936,0.9333,0.92433,0.93378,0.93159,0.92928,0.92242,0.93118,0.92192,0.93233,0.93332,0.93331,0.93694,0.92957,0.92386,0.93304,0.93187,0.93739,0.93898,0.93589,0.94688,0.94296,0.94684,0.93924,0.9379,0.93718,0.94559,0.94211,0.94844,0.94767,0.94526,0.948,0.9473,0.94967,0.93951,0.94341,0.95304,0.9496,0.94324,0.94872,0.9531,0.95074,0.94855,0.94608,0.94935,0.94231,0.94683,0.95195,0.94441,0.95738,0.94799,0.95087,0.95147,0.95112,0.9529,0.9528,0.95521,0.95454,0.95473,0.95509,0.95614,0.95207,0.95483,0.95683,0.95532,0.95473,0.95317,0.95141,0.95037,0.95771,0.96154,0.95079,0.95534,0.95368,0.95546,0.9552,0.95563,0.9544,0.9592,0.95912,0.95927,0.95991,0.95455,0.95633,0.95707,0.95509,0.95584,0.95866,0.95943,0.95852,0.95452,0.95704,0.95969,0.95768,0.96004,0.95777,0.95814,0.9579,0.96238,0.96373,0.95957,0.95838,0.96207,0.96119,0.96107,0.96491,0.96344,0.96076,0.96514,0.96638,0.95965,0.9632,0.96102,0.96407,0.96645,0.96244,0.9626,0.96226,0.96132,0.95952,0.95804,0.96273,0.96614,0.95979,0.96382,0.9609,0.96772,0.9649,0.95775,0.96128,0.95993,0.96191,0.96388,0.96365,0.96869,0.96849,0.96314,0.96574,0.96504,0.96962,0.96901,0.9636,0.95946,0.96505,0.96482,0.97004,0.96555,0.96662,0.97146,0.96883,0.96768,0.9711,0.96782,0.96766,0.96359,0.9651,0.96835,0.96808,0.96964,0.97333,0.9656,0.96709,0.96712,0.96607,0.97079,0.96855,0.97395,0.96989,0.96838,0.96194,0.9709,0.96604,0.97132,0.97152,0.9716,0.96883,0.97337,0.96976,0.97123,0.96572,0.96913,0.97278,0.97033,0.97118,0.97495,0.9695,0.97032,0.9732,0.97222,0.97197,0.97127,0.97284,0.97509,0.9724,0.97206,0.97306,0.97246,0.9707,0.97018,0.97705,0.97752,0.97404,0.97363,0.97425,0.97481,0.97231,0.97541,0.97764,0.97266,0.97377,0.97322,0.97285,0.9732,0.97348,0.97303,0.97015,0.97215,0.973,0.97836,0.97666,0.97771,0.97419,0.97604,0.97733,0.97213,0.97366,0.97308,0.97203,0.97076,0.97603,0.97542,0.97491,0.97779,0.98142,0.97573,0.97141,0.97848,0.9785,0.97317,0.9768,0.97725,0.97356,0.978,0.97831,0.97614,0.97741,0.97588,0.981,0.9725,0.97817,0.9772,0.97952,0.97924,0.98141,0.98087,0.97729,0.97739,0.97852,0.98266,0.98082,0.97568,0.98157,0.97699,0.98037,0.97957,0.98175,0.97693,0.98088,0.98125,0.97635,0.97446,0.97896,0.97897,0.98014,0.98224,0.97961,0.98189,0.97931,0.97531,0.9795,0.97618,0.9776,0.97879,0.97854,0.97455,0.97759,0.97716,0.97481,0.97681,0.97685,0.98,0.98128,0.98189,0.97723,0.97886,0.98057,0.97959,0.98012,0.97931,0.98106,0.97888,0.97835,0.97888,0.98082,0.97996,0.9829,0.97476,0.97603,0.97937,0.9817,0.98015,0.97789,0.97661,0.98164,0.97901,0.98367,0.98078,0.97945,0.97636,0.9817,0.97645,0.98061,0.98086,0.97904,0.98208,0.98063,0.97828,0.97644,0.98072,0.97746,0.97904,0.9801,0.97967,0.97854,0.98284,0.97923,0.9771,0.98293,0.98298,0.97958,0.97853,0.98399,0.98464,0.98158,0.98462,0.981,0.97499,0.9819,0.97681,0.98069,0.97868,0.97622,0.98179,0.98066,0.97822,0.98453,0.97842,0.98423,0.98132,0.97929,0.98136,0.98134,0.98411,0.97778,0.98287,0.97616,0.98046,0.98158,0.98272,0.98418,0.97896,0.97946,0.98074,0.98022,0.98495,0.9795,0.97929,0.98167,0.97882,0.98085,0.98152,0.98003,0.98111,0.98292,0.9834,0.98565,0.98247,0.98368,0.98036,0.9794,0.98214,0.98009,0.97612,0.98092,0.98134,0.98699,0.98159,0.98381,0.98039,0.98063,0.98125,0.98522,0.97651,0.98522,0.97763,0.98315,0.97693,0.98297,0.98228,0.98305,0.9845,0.98463,0.98377,0.98426,0.98326,0.97962,0.97353,0.97688,0.98163,0.9857,0.98258,0.98633,0.98338,0.9829,0.98121,0.98202,0.98542,0.98398,0.98691,0.98681,0.98338,0.98163,0.98198,0.98413,0.98444,0.98423,0.98757,0.98457,0.98382,0.98679,0.98324,0.98293,0.98423,0.98564,0.98123,0.98446,0.97974,0.97936,0.98086,0.97805,0.98254,0.98495,0.98469,0.98522,0.98508,0.9825,0.98224,0.98246,0.98418,0.98275,0.97993,0.98044,0.98333,0.98382,0.97809,0.98051,0.98033,0.98408,0.98402,0.98194,0.98077,0.98343,0.98229,0.98069,0.97976,0.98682,0.98253,0.98058,0.98232,0.98103,0.98166,0.98332,0.9801,0.98073,0.97973,0.97921,0.98287,0.98532,0.9835,0.98207,0.97982,0.98282,0.98295,0.98349,0.98214,0.98395,0.98448,0.98312,0.98914,0.98086,0.98079,0.98303,0.98629,0.98307,0.98369,0.98648,0.98511,0.98395,0.98582,0.98616,0.98546,0.98213,0.98313,0.98505,0.9829,0.98377,0.98332,0.98524,0.98651,0.98442,0.98643,0.98702,0.9826,0.98525,0.98615,0.98289,0.98586,0.98527,0.98615,0.98304,0.984,0.97909,0.9788,0.97963,0.98065,0.98303,0.9832,0.98289,0.98271,0.98647,0.98441,0.9849,0.98663,0.98494,0.98373,0.98758,0.98245,0.98496,0.98154,0.98308,0.98569,0.98431,0.98693,0.9863,0.98285,0.9838,0.98351,0.98199,0.98365,0.9838,0.98357,0.98299,0.98464,0.98752,0.98522,0.9836,0.97932,0.98666,0.98124,0.98065,0.98344,0.98487,0.98365,0.98571,0.98628,0.98603,0.98479,0.98314,0.98051,0.9829,0.97784,0.98525,0.98396,0.98541,0.98292,0.9851,0.98584,0.98269,0.98453,0.98358,0.98807,0.98414,0.9847,0.98601,0.98664,0.98543,0.98388,0.98398,0.98523,0.98077,0.98463,0.98461,0.98629,0.98209,0.98032,0.98285,0.98561,0.98567,0.98348,0.9843,0.98489,0.98329,0.9825,0.98869,0.98571,0.98484,0.9868,0.98658,0.98644,0.98457,0.98805,0.98183,0.98445,0.98403,0.98522,0.9843,0.98274,0.9857,0.98609,0.98289,0.98533,0.98303,0.984,0.98375,0.98548,0.98415,0.98718,0.98462,0.98648,0.98332,0.98418,0.98365,0.98472,0.98517,0.98202,0.98215,0.98157,0.98222,0.98419,0.98368,0.98551,0.98439,0.98603,0.98271,0.98512,0.98592,0.97986,0.98115,0.98498,0.98327,0.98316,0.98097,0.98352,0.98362,0.98318,0.98523,0.98219,0.98405,0.98258,0.98163,0.98113,0.98603,0.98363,0.98527,0.98046,0.98161,0.9817,0.98621,0.98319,0.97894,0.98247,0.98328,0.98151,0.98652,0.98316,0.98522,0.98581,0.98271,0.9836,0.98191,0.98173,0.98285,0.983,0.98526,0.98653,0.98269,0.98571,0.97993,0.98442,0.98319,0.98504,0.98398,0.98131,0.98567,0.98326,0.98643,0.98377,0.98833,0.98539,0.98229,0.98524,0.98608,0.9846,0.98581,0.98314,0.98462,0.98805,0.98427,0.98281,0.98338,0.97847,0.98465,0.98617,0.98504,0.98094,0.97933,0.98122,0.98256,0.98488,0.98659,0.98706,0.98604,0.98533,0.98654,0.98307,0.98197,0.98545,0.98249,0.98584,0.98458,0.98363,0.98372,0.98668,0.98501,0.98612,0.98813,0.98937,0.98628,0.98396,0.98599,0.98647,0.98613,0.98476,0.98431,0.98409,0.98453,0.98279,0.98606,0.98611,0.98478,0.98477,0.98304,0.98659,0.98522,0.98624,0.98725,0.98537,0.98715,0.98767,0.98416,0.98736,0.98384,0.98253,0.9812,0.983,0.98767,0.98617,0.98571,0.98704,0.9864,0.98194,0.98534,0.98499,0.98212,0.98536,0.98498,0.98592,0.98334,0.98727,0.98619,0.98531,0.9862,0.98379,0.98544,0.98589,0.98748,0.98208,0.98538,0.98571,0.98785,0.98364,0.98303,0.98693,0.98534,0.98646,0.98385,0.98594,0.98796,0.98554,0.98736,0.98535,0.98472,0.98706,0.98397,0.98804,0.98773,0.98679,0.98422,0.98808,0.98633,0.98629,0.9855,0.98541,0.9866,0.98622,0.98979,0.98701,0.98378,0.9838,0.98475,0.98622,0.98441,0.98679,0.98828,0.98621,0.98794,0.9872,0.9851,0.98541,0.9873,0.98822,0.9871,0.98857,0.98644,0.98549,0.98657,0.98443,0.98543,0.98808,0.98473,0.98461,0.98625,0.98361,0.98665,0.99082,0.98427,0.98543,0.98655,0.98465,0.99037,0.9888,0.98584,0.98449,0.98726,0.98724,0.98494,0.98568,0.98679,0.9868,0.98595,0.9849,0.9847,0.98525,0.98506,0.98581,0.9835,0.9849,0.98853,0.98888,0.98597,0.98844,0.98744,0.98519,0.98722,0.98888,0.98729,0.98463,0.98922,0.98627,0.98231,0.98616,0.98334,0.98791,0.98533,0.98703,0.98553,0.98526,0.98462,0.98492,0.98528,0.98706,0.98887,0.98786,0.98495,0.9868,0.98599,0.98649,0.98342,0.9857,0.9848,0.98834,0.98642,0.98463]
macros=[0.16643,0.19941,0.35446,0.45689,0.51379,0.53586,0.63431,0.61467,0.70676,0.66586,0.72786,0.72726,0.72831,0.73661,0.7028,0.74538,0.75045,0.75752,0.75964,0.76055,0.74898,0.77212,0.78047,0.78211,0.77905,0.77136,0.78914,0.78861,0.80095,0.79693,0.79353,0.79136,0.78762,0.796,0.79198,0.79356,0.79429,0.80863,0.7984,0.80346,0.80501,0.80858,0.81048,0.79255,0.80311,0.81863,0.81443,0.81666,0.80698,0.82045,0.8065,0.82679,0.81611,0.81847,0.82172,0.81644,0.82826,0.83033,0.82733,0.82823,0.83222,0.8245,0.82832,0.8284,0.82201,0.83237,0.82607,0.83362,0.83369,0.83332,0.83275,0.82732,0.83084,0.83743,0.84576,0.83829,0.84566,0.84141,0.84062,0.84216,0.84498,0.83956,0.84506,0.85319,0.85084,0.85611,0.85533,0.85041,0.85767,0.84268,0.85373,0.85804,0.8502,0.84859,0.84998,0.84846,0.84826,0.86105,0.8629,0.85961,0.85667,0.84817,0.85679,0.85504,0.85939,0.86945,0.86129,0.87488,0.86872,0.87318,0.86844,0.86208,0.85856,0.86783,0.87352,0.88191,0.88554,0.87081,0.87878,0.87441,0.87203,0.8696,0.88021,0.88161,0.88073,0.87114,0.8791,0.88473,0.88274,0.88166,0.87766,0.87616,0.86872,0.87871,0.88175,0.88206,0.88635,0.87585,0.88255,0.88339,0.88188,0.88084,0.88765,0.88642,0.88769,0.8872,0.88709,0.88849,0.88538,0.88797,0.88799,0.88642,0.88133,0.88834,0.8837,0.88145,0.89082,0.89444,0.8834,0.88616,0.8847,0.88908,0.88645,0.88813,0.88735,0.89144,0.89099,0.89102,0.89286,0.88597,0.88755,0.89037,0.8863,0.88854,0.8907,0.8919,0.89042,0.88452,0.88929,0.892,0.89229,0.89365,0.89177,0.88906,0.88935,0.89368,0.89698,0.89364,0.88934,0.89471,0.89513,0.89403,0.89757,0.89654,0.89326,0.89691,0.89835,0.89276,0.89545,0.89324,0.89784,0.89774,0.89674,0.90699,0.89635,0.89279,0.89559,0.89181,0.89613,0.8987,0.89524,0.8966,0.8961,0.90093,0.89685,0.89373,0.89684,0.89485,0.89337,0.89673,0.89821,0.9015,0.89882,0.89384,0.89898,0.89797,0.90218,0.90198,0.89421,0.88916,0.89821,0.89727,0.9013,0.89901,0.8987,0.91471,0.90209,0.91109,0.90425,0.90018,0.90388,0.89761,0.89748,0.90052,0.90054,0.90204,0.90377,0.89812,0.90038,0.89887,0.89776,0.90228,0.90148,0.90592,0.9142,0.90142,0.89687,0.90176,0.90041,0.90392,0.91404,0.90433,0.90236,0.92187,0.9024,0.9039,0.90137,0.91505,0.90637,0.90295,0.90547,0.91091,0.90333,0.90297,0.91754,0.90898,0.90921,0.91708,0.90973,0.91358,0.90973,0.90712,0.90926,0.90905,0.9059,0.90472,0.92629,0.91625,0.90896,0.91992,0.91385,0.91163,0.92065,0.91239,0.92572,0.90853,0.91023,0.9094,0.90851,0.91065,0.90921,0.90692,0.90429,0.90757,0.90916,0.91919,0.92335,0.91738,0.90961,0.91325,0.91745,0.91347,0.924,0.91131,0.9089,0.90801,0.91314,0.91308,0.91076,0.91967,0.92365,0.92559,0.90844,0.92519,0.91798,0.91092,0.91894,0.92493,0.91865,0.91783,0.91831,0.92006,0.92693,0.92767,0.91887,0.92811,0.9153,0.92499,0.91707,0.91657,0.93063,0.92069,0.91634,0.91716,0.93113,0.93552,0.93885,0.92302,0.92061,0.9176,0.93036,0.91826,0.92238,0.91838,0.9213,0.92255,0.91315,0.91149,0.91396,0.91685,0.93093,0.92126,0.92816,0.92133,0.93686,0.92731,0.92033,0.91625,0.92202,0.91671,0.91617,0.91227,0.91664,0.92636,0.91363,0.91939,0.92322,0.92624,0.92982,0.92135,0.9299,0.9296,0.92183,0.92671,0.92475,0.92756,0.91702,0.92024,0.92975,0.91927,0.93382,0.93339,0.92864,0.91542,0.91472,0.93956,0.93981,0.94085,0.92089,0.9279,0.92983,0.92761,0.93187,0.91978,0.9275,0.91764,0.92997,0.93394,0.93049,0.92888,0.93659,0.93215,0.9312,0.92983,0.92521,0.92425,0.91467,0.91794,0.92248,0.92681,0.91555,0.922,0.92421,0.93043,0.92302,0.92217,0.92654,0.92528,0.9218,0.93565,0.93069,0.93312,0.93793,0.91178,0.92996,0.93709,0.9323,0.93778,0.92383,0.93016,0.92989,0.92348,0.92338,0.93635,0.93837,0.94459,0.94402,0.91889,0.93107,0.94683,0.92667,0.93072,0.91759,0.93063,0.94211,0.93283,0.94711,0.92557,0.92961,0.93384,0.91842,0.94144,0.93427,0.93004,0.93528,0.91964,0.93458,0.9301,0.93192,0.92682,0.93341,0.93363,0.94122,0.93292,0.94315,0.9157,0.93091,0.93571,0.93883,0.91899,0.92427,0.93452,0.94539,0.92378,0.93913,0.91788,0.92954,0.91768,0.94429,0.91101,0.94675,0.9303,0.92323,0.91582,0.94194,0.92191,0.92257,0.94361,0.93712,0.92394,0.94083,0.92256,0.93717,0.91581,0.91717,0.92552,0.9475,0.92545,0.94769,0.93509,0.93534,0.93308,0.93212,0.93398,0.94344,0.9438,0.94493,0.92325,0.92186,0.94024,0.93326,0.94254,0.93303,0.94867,0.93349,0.94291,0.95436,0.94578,0.93578,0.9373,0.95098,0.93213,0.94643,0.9323,0.93621,0.94044,0.92201,0.94116,0.94453,0.93288,0.94537,0.9476,0.94633,0.93489,0.93733,0.95332,0.93581,0.93033,0.93895,0.94595,0.94372,0.92877,0.94018,0.93104,0.95114,0.93711,0.94529,0.93728,0.9271,0.92432,0.93582,0.9196,0.94892,0.93183,0.92167,0.94037,0.93957,0.9389,0.94113,0.92034,0.94057,0.93201,0.91982,0.93552,0.94366,0.94504,0.92259,0.93121,0.946,0.92612,0.94351,0.93585,0.92652,0.94309,0.93934,0.9502,0.94201,0.9208,0.92188,0.94817,0.94305,0.94623,0.94784,0.93816,0.94368,0.9449,0.95157,0.94761,0.95738,0.94321,0.94391,0.92273,0.94322,0.94217,0.9545,0.94514,0.94319,0.94388,0.95166,0.92573,0.92791,0.94798,0.94603,0.94762,0.94473,0.94123,0.93617,0.93724,0.92864,0.93212,0.92959,0.9226,0.93575,0.94373,0.9423,0.93232,0.94808,0.95421,0.93439,0.94478,0.94416,0.94359,0.96644,0.9214,0.94459,0.92202,0.92324,0.94626,0.94974,0.95507,0.94846,0.92818,0.94297,0.94686,0.92235,0.93581,0.94274,0.93237,0.94853,0.95039,0.94638,0.94142,0.94935,0.91659,0.94133,0.93214,0.92003,0.9395,0.94081,0.93719,0.9455,0.94864,0.94836,0.93713,0.93883,0.91964,0.93272,0.9273,0.92371,0.92594,0.94399,0.93303,0.93843,0.9453,0.93137,0.94841,0.95765,0.95652,0.94414,0.95008,0.94488,0.95113,0.92459,0.94565,0.94372,0.93401,0.94411,0.94438,0.94389,0.93125,0.93211,0.93897,0.93597,0.94884,0.95038,0.9426,0.94389,0.93877,0.93239,0.92602,0.93651,0.94787,0.92926,0.94568,0.93757,0.94755,0.93023,0.94987,0.92422,0.94877,0.92299,0.94792,0.94357,0.93602,0.93487,0.94917,0.94163,0.95027,0.943,0.92363,0.94635,0.95062,0.95395,0.94804,0.94812,0.94477,0.93493,0.9405,0.93243,0.93372,0.94716,0.92088,0.93285,0.91789,0.946,0.94469,0.93278,0.94772,0.94729,0.95127,0.92214,0.93445,0.95084,0.92885,0.92626,0.93841,0.95501,0.94505,0.93216,0.93535,0.94851,0.95535,0.94307,0.93781,0.93804,0.94186,0.94106,0.93939,0.95798,0.94858,0.92306,0.94086,0.94107,0.92754,0.96432,0.93358,0.92975,0.92121,0.95217,0.93452,0.95796,0.95567,0.94792,0.9585,0.93151,0.95539,0.93465,0.92429,0.92601,0.95297,0.94692,0.95404,0.94149,0.95686,0.93427,0.94692,0.92215,0.93639,0.94913,0.93415,0.94414,0.94499,0.94085,0.92249,0.96342,0.93996,0.93552,0.92739,0.94512,0.94339,0.9481,0.93856,0.92791,0.94897,0.93961,0.92487,0.94287,0.92988,0.93396,0.93844,0.94029,0.93015,0.93247,0.94202,0.94175,0.94306,0.94473,0.94509,0.95066,0.94728,0.95143,0.94545,0.92445,0.94773,0.94048,0.93563,0.93989,0.93117,0.92369,0.93811,0.94706,0.9411,0.95516,0.95707,0.94144,0.94346,0.94769,0.93873,0.93876,0.94677,0.94745,0.93891,0.94358,0.94898,0.94466,0.95088,0.93726,0.93698,0.94541,0.94827,0.93475,0.94726,0.92827,0.94696,0.95426,0.94816,0.94668,0.94413,0.9434,0.94226,0.93251,0.93621,0.95561,0.93886,0.9455,0.95358,0.9521,0.94209,0.94108,0.93464,0.92106,0.95,0.93763,0.95104,0.93601,0.94235,0.9245,0.944,0.93517,0.92254,0.94468,0.94791,0.95242,0.935,0.93797,0.94411,0.95165,0.93184,0.93288,0.93732,0.94504,0.95143,0.93346,0.95419,0.95274,0.94734,0.95508,0.95367,0.94891,0.94478,0.93693,0.96528,0.946,0.94883,0.94536,0.94919,0.94879,0.94828,0.94741,0.93524,0.94461,0.95416,0.96337,0.9553,0.94368,0.92274,0.94636,0.94847,0.93669,0.93832,0.94986,0.95392,0.93917,0.94501,0.95351,0.94737,0.94202,0.95318,0.952,0.95565,0.95181,0.94448,0.95829,0.94397,0.94772,0.95183,0.94741,0.9375,0.94316,0.93228,0.95587,0.96089,0.93376,0.94466,0.94779,0.92427,0.95041,0.95922,0.93496,0.94496,0.9559,0.95218,0.94753,0.9413,0.94854,0.94873,0.95895,0.93966,0.9371,0.95795,0.94335,0.9448,0.92152,0.94381,0.96256,0.95239,0.94446,0.96254,0.96136,0.93696,0.93888,0.9571,0.94616,0.94758,0.94659,0.95401,0.92218,0.94719,0.93222,0.9361,0.94014,0.94826,0.94372,0.94501,0.93675,0.9404,0.94651,0.94535,0.95643,0.95245,0.93939,0.94827,0.94786,0.94819,0.92302,0.95784,0.94416,0.9629,0.95473,0.94435]
times=[1.57196,1.00033,0.75901,0.62292,0.53454,0.47212,0.42556,0.38926,0.36012,0.33618,0.31612,0.29906,0.28437,0.27157,0.26031,0.25033,0.2414,0.23337,0.2261,0.21949,0.21345,0.20791,0.20281,0.19809,0.19372,0.18966,0.18587,0.18233,0.17901,0.17589,0.17296,0.17019,0.16757,0.1651,0.16275,0.16052,0.15841,0.15639,0.15447,0.15263,0.15088,0.1492,0.14759,0.14605,0.14457,0.14315,0.14179,0.14048,0.13922,0.138,0.13683,0.1357,0.13461,0.13355,0.13254,0.13155,0.1306,0.12968,0.12878,0.12792,0.12708,0.12626,0.12547,0.1247,0.12396,0.12323,0.12253,0.12184,0.12117,0.12052,0.11989,0.11928,0.11868,0.11811,0.11756,0.11703,0.11651,0.116,0.11551,0.11503,0.11457,0.11412,0.11367,0.11324,0.11282,0.11242,0.11202,0.11164,0.11126,0.1109,0.11055,0.1102,0.10986,0.10952,0.1092,0.10887,0.10856,0.10825,0.10794,0.10764,0.10735,0.10706,0.10677,0.10649,0.10621,0.10594,0.10567,0.10541,0.10515,0.1049,0.10465,0.1044,0.10416,0.10392,0.10369,0.10346,0.10323,0.10301,0.10279,0.10258,0.10237,0.10216,0.10195,0.10175,0.10155,0.10135,0.10116,0.10097,0.10078,0.10059,0.10041,0.10023,0.10006,0.09988,0.09971,0.09954,0.09937,0.09921,0.09905,0.09889,0.09873,0.09857,0.09841,0.09826,0.09811,0.09796,0.09782,0.09767,0.09753,0.09739,0.09725,0.09711,0.09698,0.09685,0.09671,0.09658,0.09645,0.09633,0.0962,0.09608,0.09595,0.09583,0.09571,0.0956,0.09548,0.09536,0.09525,0.09514,0.09503,0.09492,0.09481,0.0947,0.09459,0.09449,0.09438,0.09428,0.09418,0.09408,0.09398,0.09388,0.09378,0.09368,0.09359,0.09349,0.0934,0.09331,0.09322,0.09313,0.09304,0.09295,0.09286,0.09277,0.09268,0.0926,0.09251,0.09243,0.09234,0.09226,0.09218,0.0921,0.09202,0.09193,0.09186,0.09178,0.0917,0.09162,0.09154,0.09147,0.09139,0.09132,0.09124,0.09117,0.0911,0.09102,0.09095,0.09088,0.09081,0.09074,0.09067,0.0906,0.09053,0.09047,0.0904,0.09033,0.09026,0.0902,0.09013,0.09007,0.09,0.08994,0.08988,0.08982,0.08975,0.08969,0.08963,0.08957,0.08951,0.08945,0.08939,0.08933,0.08927,0.08921,0.08916,0.0891,0.08904,0.08899,0.08893,0.08888,0.08882,0.08877,0.08871,0.08866,0.0886,0.08855,0.0885,0.08844,0.08839,0.08834,0.08829,0.08824,0.08819,0.08814,0.08809,0.08804,0.08799,0.08794,0.08789,0.08784,0.08779,0.08774,0.0877,0.08765,0.0876,0.08755,0.08751,0.08746,0.08741,0.08737,0.08732,0.08728,0.08723,0.08719,0.08715,0.0871,0.08706,0.08702,0.08697,0.08693,0.08689,0.08685,0.08681,0.08677,0.08673,0.08669,0.08665,0.08661,0.08657,0.08654,0.0865,0.08646,0.08643,0.08639,0.08635,0.08632,0.08628,0.08624,0.08621,0.08617,0.08614,0.0861,0.08607,0.08603,0.086,0.08596,0.08593,0.08589,0.08586,0.08583,0.08579,0.08576,0.08573,0.08569,0.08566,0.08563,0.08559,0.08556,0.08553,0.0855,0.08546,0.08543,0.0854,0.08537,0.08534,0.08531,0.08528,0.08525,0.08522,0.08519,0.08516,0.08513,0.0851,0.08507,0.08504,0.08501,0.08498,0.08495,0.08492,0.0849,0.08487,0.08484,0.08481,0.08479,0.08476,0.08473,0.08471,0.08468,0.08465,0.08463,0.0846,0.08458,0.08455,0.08452,0.0845,0.08448,0.08445,0.08443,0.0844,0.08438,0.08435,0.08433,0.08431,0.08428,0.08426,0.08423,0.08421,0.08419,0.08416,0.08414,0.08412,0.08409,0.08407,0.08405,0.08402,0.084,0.08398,0.08396,0.08393,0.08391,0.08389,0.08386,0.08384,0.08382,0.0838,0.08378,0.08375,0.08373,0.08371,0.08369,0.08367,0.08364,0.08362,0.0836,0.08358,0.08356,0.08354,0.08352,0.08349,0.08347,0.08345,0.08343,0.08341,0.08339,0.08337,0.08335,0.08333,0.08331,0.08329,0.08327,0.08325,0.08322,0.0832,0.08318,0.08316,0.08314,0.08312,0.0831,0.08308,0.08306,0.08304,0.08302,0.083,0.08298,0.08296,0.08295,0.08293,0.08291,0.08289,0.08287,0.08285,0.08283,0.08281,0.08279,0.08277,0.08275,0.08273,0.08271,0.0827,0.08268,0.08266,0.08264,0.08262,0.0826,0.08258,0.08257,0.08255,0.08253,0.08251,0.08249,0.08247,0.08246,0.08244,0.08242,0.0824,0.08238,0.08237,0.08235,0.08233,0.08231,0.0823,0.08228,0.08226,0.08224,0.08223,0.08221,0.08219,0.08217,0.08216,0.08214,0.08212,0.08211,0.08209,0.08207,0.08205,0.08204,0.08202,0.082,0.08199,0.08197,0.08196,0.08194,0.08192,0.08191,0.08189,0.08187,0.08186,0.08184,0.08183,0.08181,0.08179,0.08178,0.08176,0.08175,0.08173,0.08172,0.0817,0.08168,0.08167,0.08165,0.08164,0.08162,0.08161,0.08159,0.08158,0.08156,0.08155,0.08153,0.08152,0.08151,0.08149,0.08148,0.08146,0.08145,0.08143,0.08142,0.0814,0.08139,0.08137,0.08136,0.08135,0.08133,0.08132,0.0813,0.08129,0.08128,0.08126,0.08125,0.08123,0.08122,0.08121,0.08119,0.08118,0.08116,0.08115,0.08114,0.08112,0.08111,0.0811,0.08109,0.08107,0.08106,0.08105,0.08103,0.08102,0.08101,0.081,0.08098,0.08097,0.08096,0.08095,0.08094,0.08092,0.08091,0.0809,0.08089,0.08088,0.08086,0.08085,0.08084,0.08083,0.08082,0.08081,0.0808,0.08078,0.08077,0.08076,0.08075,0.08074,0.08073,0.08072,0.08071,0.0807,0.08069,0.08067,0.08066,0.08065,0.08064,0.08063,0.08062,0.08061,0.0806,0.08059,0.08058,0.08057,0.08056,0.08055,0.08054,0.08053,0.08052,0.08051,0.0805,0.08049,0.08048,0.08047,0.08046,0.08045,0.08044,0.08043,0.08042,0.08041,0.0804,0.08039,0.08038,0.08037,0.08036,0.08035,0.08034,0.08033,0.08032,0.08031,0.0803,0.08029,0.08028,0.08027,0.08026,0.08025,0.08024,0.08024,0.08023,0.08022,0.08021,0.0802,0.08019,0.08018,0.08017,0.08016,0.08015,0.08014,0.08014,0.08013,0.08012,0.08011,0.0801,0.08009,0.08008,0.08007,0.08007,0.08006,0.08005,0.08004,0.08003,0.08002,0.08001,0.08,0.08,0.07999,0.07998,0.07997,0.07996,0.07995,0.07995,0.07994,0.07993,0.07992,0.07991,0.0799,0.0799,0.07989,0.07988,0.07987,0.07986,0.07985,0.07985,0.07984,0.07983,0.07982,0.07981,0.07981,0.0798,0.07979,0.07978,0.07977,0.07977,0.07976,0.07975,0.07974,0.07973,0.07973,0.07972,0.07971,0.0797,0.07969,0.07969,0.07968,0.07967,0.07966,0.07966,0.07965,0.07964,0.07963,0.07962,0.07962,0.07961,0.0796,0.07959,0.07959,0.07958,0.07957,0.07956,0.07956,0.07955,0.07954,0.07953,0.07953,0.07952,0.07951,0.0795,0.0795,0.07949,0.07948,0.07948,0.07947,0.07946,0.07945,0.07945,0.07944,0.07943,0.07943,0.07942,0.07941,0.0794,0.0794,0.07939,0.07938,0.07938,0.07937,0.07936,0.07935,0.07935,0.07934,0.07933,0.07933,0.07932,0.07931,0.07931,0.0793,0.07929,0.07929,0.07928,0.07927,0.07926,0.07926,0.07925,0.07924,0.07924,0.07923,0.07922,0.07922,0.07921,0.0792,0.0792,0.07919,0.07919,0.07918,0.07917,0.07917,0.07916,0.07915,0.07915,0.07914,0.07913,0.07913,0.07912,0.07911,0.07911,0.0791,0.0791,0.07909,0.07908,0.07908,0.07907,0.07906,0.07906,0.07905,0.07905,0.07904,0.07903,0.07903,0.07902,0.07902,0.07901,0.079,0.079,0.07899,0.07899,0.07898,0.07897,0.07897,0.07896,0.07896,0.07895,0.07895,0.07894,0.07893,0.07893,0.07892,0.07892,0.07891,0.0789,0.0789,0.07889,0.07889,0.07888,0.07888,0.07887,0.07887,0.07886,0.07885,0.07885,0.07884,0.07884,0.07883,0.07883,0.07882,0.07882,0.07881,0.07881,0.0788,0.0788,0.07879,0.07878,0.07878,0.07877,0.07877,0.07876,0.07876,0.07875,0.07875,0.07874,0.07874,0.07873,0.07873,0.07872,0.07872,0.07871,0.07871,0.0787,0.0787,0.07869,0.07869,0.07868,0.07868,0.07867,0.07867,0.07866,0.07866,0.07865,0.07865,0.07864,0.07864,0.07863,0.07863,0.07862,0.07862,0.07861,0.07861,0.0786,0.0786,0.07859,0.07859,0.07858,0.07858,0.07857,0.07857,0.07857,0.07856,0.07856,0.07855,0.07855,0.07854,0.07854,0.07853,0.07853,0.07852,0.07852,0.07851,0.07851,0.0785,0.0785,0.07849,0.07849,0.07848,0.07848,0.07848,0.07847,0.07847,0.07846,0.07846,0.07845,0.07845,0.07844,0.07844,0.07844,0.07843,0.07843,0.07842,0.07842,0.07841,0.07841,0.0784,0.0784,0.0784,0.07839,0.07839,0.07838,0.07838,0.07837,0.07837,0.07837,0.07836,0.07836,0.07835,0.07835,0.07834,0.07834,0.07834,0.07833,0.07833,0.07832,0.07832,0.07831,0.07831,0.07831,0.0783,0.0783,0.07829,0.07829,0.07828,0.07828,0.07828,0.07827,0.07827,0.07826,0.07826,0.07825,0.07825,0.07825,0.07824,0.07824,0.07823,0.07823,0.07823,0.07822,0.07822,0.07821,0.07821,0.0782,0.0782,0.0782,0.07819,0.07819,0.07818,0.07818,0.07818,0.07817,0.07817,0.07816,0.07816,0.07816,0.07815,0.07815,0.07814,0.07814,0.07814,0.07813,0.07813,0.07812,0.07812,0.07811,0.07811,0.07811,0.0781,0.0781,0.07809,0.07809,0.07809,0.07808,0.07808,0.07808,0.07807,0.07807,0.07806,0.07806,0.07806,0.07805,0.07805,0.07804,0.07804,0.07804,0.07803,0.07803]

loss_stream = hv.streams.Buffer(get_loss_data())
time_stream = hv.streams.Buffer(get_time_data())
micro_stream = hv.streams.Buffer(get_micro_data())
macro_stream = hv.streams.Buffer(get_macro_data())

loss_dmap = hv.DynamicMap(loss_box, streams=[loss_stream])
time_dmap = hv.DynamicMap(time_box, streams=[time_stream])
micro_dmap = hv.DynamicMap(micro_box, streams=[micro_stream])
macro_dmap = hv.DynamicMap(macro_box, streams=[macro_stream])

plot1 = (loss_dmap + time_dmap).opts(opts.Curve(height=400, width=400,show_grid=True), opts.Curve(height=400, width=400,show_grid=True))
plot2 = (micro_dmap + macro_dmap).opts(opts.Curve(height=400, width=400,show_grid=True),opts.Curve(height=400, width=400,show_grid=True))

# Render plot and attach periodic callback
doc1 = renderer.server_doc(plot1)
doc2 = renderer.server_doc(plot2)

if i>=0:

	doc1.add_periodic_callback(cb_loss_time, 10)
	doc2.add_periodic_callback(cb_micro_macro, 10)
	i=i+1
	doc1.show()





