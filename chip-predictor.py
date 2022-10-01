import argparse
import yaml
from pprint import pprint
import copy
import math
import csv


def vague_get(dic_in, key_name):
    start_check_duplicate = False
    for each in dic_in:
        if key_name in each and not start_check_duplicate:
            key_found = each
            start_check_duplicate = True
        elif key_name in each and start_check_duplicate:
            assert False, "[{k0}] is ambiguous: [{k1}] and [{k2}] are found".format(k0=key_name, k1=key_found, k2 = each)
    if not start_check_duplicate:
        print (dic_in.keys())
        assert False, "{k0} is not found".format(k0=key_name)
    return dic_in[key_found]

class dict_with_vague_get(object):
    def __init__(self, dic):
        self.data = dic
    def get(self, vague_key_list):
        cur = self.data
        for vague_key in vague_key_list:
            cur = vague_get(cur, vague_key)
            # print (cur)
        return cur


def yaml_read(fname):
    with open(fname, 'r') as yaml_in:
        json_obj = yaml.safe_load(yaml_in) # yaml_object will be a list or a dict
    print ('')
    return dict_with_vague_get(json_obj)


class convPrediction(object):
    def __init__(self, hw_yaml, df_yaml, layer_yaml, pred_in = None) -> None:
        if pred_in is None:
            self.hw = yaml_read(hw_yaml)
            self.df = yaml_read(df_yaml)
            self.l = yaml_read(layer_yaml)
            self.pred_in = {}
            self.__parse_layer_info()
            self.__parse_df_info()
            self.__parse_hw_info()
            # pprint (self.pred_in)
        else:
            self.pred_in = copy.deepcopy(pred_in)
        # init values that will be upated in self.run()
        self.energy = { i: {j:0 for j in ['I', 'O', 'W']}
                        for i in [ 'dram', 'sram', 'noc', 'comp', 'rf' ]}
        self.latency = { i: {j:0 for j in ['I', 'O', 'W']}
                        for i in [ 'sram', 'noc', 'comp']} # the unit for latency: cycles (not seconds)
        self.buffer_size_req = { i: {j:0 for j in ['I', 'O', 'W']}
                        for i in [ 'sram', 'rf' ]} # required buffer size
        self.access = { i: {j:0 for j in ['I', 'O', 'W']}
                        for i in [ 'dram', 'sram', 'noc_forward', 'noc_unicast', 'noc_multicast',  'rf' ]}
        self.active_pe_rows = 0
        self.active_pe_cols = 0
        self.comp = 0
        # init values that will be upated in self.run()
        self.weight_loops = ['C', 'K', 'R', 'S']
        self.input_loops = ['N', 'C', 'Xo', 'Yo', 'R', 'S']
        self.output_loops = ['N', 'K', 'Xo', 'Yo']
        self.__calc_dram_loop_orders()
        self.__calc_sram_loop_orders()
        self.__calc_noc_loop_orders()
        self.__calc_rf_loop_orders()
    def __dic_create(self, dic_template):
        ret = {}
        for each in dic_template:
            key_list = [each] if isinstance(each, str) else each
            cur_lvl = ret
            for key in key_list:
                if key not in cur_lvl:
                    cur_lvl[key] = {}
                cur_lvl = cur_lvl[key]
        return ret
    def __dic_leaf(self, dic_in, list_in):
        cur = dic_in
        for each in list_in[:-1]:
            cur = cur[each]
        return cur
    def __parse_info(self, raw, parse_template, category):
        parsed = self.__dic_create(dic_template=parse_template)
        # print (parsed)
        for each in parse_template:
            key_list = [each] if isinstance(each, str) else each
            dic_to_update = self.__dic_leaf(dic_in=parsed, list_in=key_list)
            if ('size' in key_list or 'bw' in key_list) and raw.get(key_list)==-1:
                dic_to_update[key_list[-1]] = float('inf')
            elif 'prec' in key_list:
                dic_to_update[key_list[-1]] = raw.get(key_list) / 8.0
            else:
                dic_to_update[key_list[-1]] = raw.get(key_list) 
        self.pred_in.update({category: parsed})
    def __parse_layer_info(self):
        l = self.l
        parse_template = ['N', 'C', 'Yo', 'Xo', 'K', 'R', 'S', 'U'] \
                       + [['prec'] + [each] for each in ['A', 'P', 'W']] 
        self.__parse_info(raw=l, parse_template=parse_template, category='l')
    def __parse_df_info(self):
        df = self.df
        loop_orders = [ ['order'] + [each]
                        for each in ['dram', 'noc', 'rf', 'sram'] ]
        loop_tilings = [ ['tiling', i] + [j]
                         for i in df.get(['tiling'])
                         for j in df.get(['tiling', i]) if i!='noc' ] \
                     + [ ['tiling', 'noc'] + [j] + [k]
                         for j in df.get(['tiling', 'noc'])
                         for k in ['map', 'val']]
        min_refresh = [['ref'] + [i] for i in ['rf', 'sram'] ]
        parse_template = loop_orders + loop_tilings + min_refresh
        self.__parse_info(raw=df, parse_template=parse_template, category='df')
    def __parse_hw_info(self):
        hw = self.hw
        hw_noc = [ ['noc'] + [i]
                   for i in ['H', 'W'] ] \
               + [ ['noc'] + [i, j]
                   for i in ['forward', 'multicast', 'unicast']
                   for j in ['bw', 'energy', 'I', 'O', 'W'] ]
        hw_mem = [ [i] + ['alloc'] + [j]
                   for i in ['rf', 'sram']
                   for j in ['I', 'O', 'W'] ] \
               + [ [i] + [j] + [k]
                   for i in ['rf', 'sram']
                   for j in [i+str(idx) for idx in range(3)]
                   for k in ['bw', 'size', 'energy'] ]
        parse_template = hw_noc + [['pe', 'energy']] + [['dram', 'energy']] + hw_mem
        self.__parse_info(raw=hw, parse_template=parse_template, category='hw')
    def __get_loop_bound(self, loop_name):
        # return the prod of all levels as well as the prod at NoC level only
        tilings = self.pred_in['df']['tiling']
        to_prod_non_noc = [ tilings[hierarchy][loop]
                            for hierarchy in tilings
                            for loop in tilings[hierarchy]
                            if loop == loop_name and isinstance(tilings[hierarchy][loop], int ) ]
        to_prod_noc = [ tilings['noc'][loop]['val']
                        for loop in tilings['noc']
                        if loop == loop_name ]
        return math.prod(to_prod_noc+to_prod_non_noc)
    def __check_loop_bound(self, loop_name):
        loop_bound_prod = self.__get_loop_bound(loop_name)
        loop_bound_req = self.pred_in['l'][loop_name]
        assert loop_bound_prod >= loop_bound_req, 'For the loop {l}, loop bound {l1} is smaller than the minimum required value {l2}'.format(l=loop_name, l1=loop_bound_prod, l2 =  loop_bound_req)
    def __check_loop_bounds(self):
        for loop_name in self.pred_in['l']:
            if loop_name in [ 'prec', 'U' ]:
                continue
            self.__check_loop_bound(loop_name)
    def __check_RS_loops(self):
        # no R or S dram for loops
        assert 'R' not in self.dram_loops, 'loop R cannot be in the dram loops {loops}'.format(loops=str(self.dram_loops))
        assert 'S' not in self.dram_loops, 'loop S cannot be in the dram loops {loops}'.format(loops=str(self.dram_loops))
    def __check_pe_util(self):
        self.calc_active_pe()
        noc_info = self.pred_in['hw']['noc']
        assert self.active_pe_rows <= noc_info['H'], 'active PE rows ({r}) cannot be larger than noc height ({H})'.format(r=self.active_pe_rows, H = noc_info['H'])
        assert self.active_pe_cols <= noc_info['W'], 'active PE cols ({r}) cannot be larger than noc width ({W})'.format(r=self.active_pe_cols, H = noc_info['W'])
    def __check_buf_size_singly(self, key1, key2, buf_size_req):
        tlb = {'I': 'input', 'O': 'output', 'W': 'weight'}
        self.buffer_size_req[key1][key2] = buf_size_req
        buf_size = self.get_buffer_size(key1,key2)
        assert buf_size_req <= buf_size, \
            'required {k1} size of {k2} ({v1} byte) should not be larger than the actual {k1} size ({v2} byte)' \
            .format(v1=buf_size_req, v2= buf_size, k1=key1, k2 = tlb[key2])
    def __check_buf_size_all(self):
        tlb = {'I': 'input', 'O': 'output', 'W': 'weight'}
        buf_sizes_req = {i+str(j):0 for i in ['rf','sram'] for j in range(3)}
        buf_sizes = {i+str(j):self.pred_in['hw'][i][i+str(j)]['size'] for i in ['rf','sram'] for j in range(3)}
        # pprint (buf_sizes_req)
        # pprint (buf_sizes)
        for key1 in self.buffer_size_req:
            dict_each = self.buffer_size_req[key1]
            for key2 in dict_each:
                buf_sizes_req[ self.pred_in['hw'][key1]['alloc'][key2] ] += self.buffer_size_req[key1][key2]
        for each in buf_sizes_req:
            assert buf_sizes_req[each] <= buf_sizes[each], \
                'required {k} size ({v1} byte) should not be larger than the actual {k} size ({v2} byte)' \
                .format(v1=buf_sizes_req[each], v2= buf_sizes[each], k=each)
    def __calc_dram_loop_orders(self):
        self.dram_loops = self.pred_in['df']['order']['dram'].split(',')
        self.dram_loops_no_ref = [loop_name for loop_name in self.dram_loops if 'refresh' not in loop_name]
    def __calc_sram_loop_orders(self):
        self.sram_loops = self.pred_in['df']['order']['sram'].split(',')
        self.sram_loops_no_ref = [loop_name for loop_name in self.sram_loops if 'refresh' not in loop_name]
    def __calc_noc_loop_orders(self):
        self.noc_loops = self.pred_in['df']['order']['noc'].split(',')
        self.noc_loops_no_ref = [loop_name for loop_name in self.noc_loops if 'refresh' not in loop_name]
    def __calc_rf_loop_orders(self):
        self.rf_loops = self.pred_in['df']['order']['rf'].split(',')
        self.rf_loops_no_ref = [loop_name for loop_name in self.rf_loops if 'refresh' not in loop_name]
    def calc_active_pe(self):
        noc_tilings = self.pred_in['df']['tiling']['noc']
        H_to_prod = [ noc_tilings[loop]['val'] for loop in noc_tilings if noc_tilings[loop]['val'] == 'H' ]
        W_to_prod = [ noc_tilings[loop]['val'] for loop in noc_tilings if noc_tilings[loop]['val'] == 'W' ]
        self.active_pe_rows = math.prod(H_to_prod)
        self.active_pe_cols = math.prod(W_to_prod)
    def __split_loops_by_refresh(self, loops_in, refresh_name):
        ref_loc = len(loops_in) if refresh_name not in loops_in else loops_in.index(refresh_name)
        loops_below_refresh = [each for each in loops_in[ref_loc+1:] if 'refresh' not in each]
        loops_above_refresh = [each for each in loops_in[:ref_loc] if 'refresh' not in each]
        return loops_below_refresh, loops_above_refresh
    def __input_sram_refresh_amount(self, sram_loops_above_refresh_loc, sram_loops_below_refresh_loc):
        tilings = self.pred_in['df']['tiling']
        val_to_prod_sram = [ tilings['sram'][loop_name]
                             for loop_name in sram_loops_below_refresh_loc
                             if loop_name in self.input_loops and loop_name not in ['Xo', 'Yo', 'R', 'S'] ]
        val_to_prod_noc = [ tilings['noc'][loop_name]['val']
                            for loop_name in self.noc_loops_no_ref
                            if loop_name in self.input_loops and loop_name not in ['Xo', 'Yo', 'R', 'S'] ]
        val_to_prod_rf = [ tilings['rf'][loop_name]
                           for loop_name in self.rf_loops_no_ref
                           if loop_name in self.input_loops and loop_name not in ['Xo', 'Yo', 'R', 'S'] ]                             
        Xo_sram = tilings['sram']['Xo'] if 'Xo' in sram_loops_below_refresh_loc else 1
        Yo_sram = tilings['sram']['Yo'] if 'Yo' in sram_loops_below_refresh_loc else 1
        Xo_noc = tilings['noc']['Xo']['val'] if 'Xo' in self.noc_loops_no_ref else 1
        Yo_noc = tilings['noc']['Yo']['val'] if 'Yo' in self.noc_loops_no_ref else 1
        Xo_rf = tilings['rf']['Xo'] if 'Xo' in self.rf_loops_no_ref else 1
        Yo_rf = tilings['rf']['Yo'] if 'Yo' in self.rf_loops_no_ref else 1
        Xi = (Xo_sram*Xo_noc*Xo_rf-1)*self.pred_in['l']['U'] + self.pred_in['l']['S']
        Yi = (Yo_sram*Yo_noc*Yo_rf-1)*self.pred_in['l']['U'] + self.pred_in['l']['R']
        prod_non_XY = math.prod(val_to_prod_sram + val_to_prod_noc + val_to_prod_rf)
        data_amount_init_load = prod_non_XY * Xi * Yi 
        closest_loop_sram = self.__find_closest_related_above(loops=sram_loops_above_refresh_loc, related_loops= ['Xo', 'Yo', 'R', 'S'])
        closest_loop_dram = self.__find_closest_related_above(loops=self.dram_loops_no_ref, related_loops = ['Xo', 'Yo', 'R', 'S'])
        NX, NY, NS, NR, NSR = (False, False, False, False, False)
        if not self.pred_in['df']['ref']['sram'] or (closest_loop_sram == '' and closest_loop_dram == ''):# minumum refresh not enabled or related loop not found
            data_amount_refresh = data_amount_init_load
        elif (closest_loop_sram == '' and closest_loop_dram == 'Xo') or closest_loop_sram == 'Xo':
            assert 'S' in sram_loops_below_refresh_loc or 'S' in self.rf_loops_no_ref or 'S' in self.noc_loops_no_ref, 'when calculating input sram refresh volume: Xo found above refresh loc but no S found below refresh loc'
            Xi_new = Xo_sram*Xo_noc*Xo_rf *  self.pred_in['l']['U']
            data_amount_refresh = prod_non_XY * Xi_new * Yi
            NX = True
        elif (closest_loop_sram == '' and closest_loop_dram == 'Yo') or closest_loop_sram == 'Yo':
            assert 'R' in sram_loops_below_refresh_loc or 'R' in self.rf_loops_no_ref or 'R' in self.noc_loops_no_ref, 'when calculating input sram refresh volume: Yo found above refresh loc but no R found below refresh loc'
            Yi_new = Yo_sram*Yo_noc*Yo_rf * self.pred_in['l']['U']
            data_amount_refresh = prod_non_XY * Xi* Yi_new
            NY = True
        elif (closest_loop_sram == '' and closest_loop_dram in ['R', 'S']):
            idx = self.dram_loops_no_ref.index(closest_loop_dram)
            Xi_new = max(self.pred_in['l']['U'], (Xo_sram*Xo_noc*Xo_rf-1)*self.pred_in['l']['U'] )
            Yi_new = max(self.pred_in['l']['U'], (Yo_sram*Yo_noc*Yo_rf-1)*self.pred_in['l']['U'] )
            if idx == 0 or set([self.dram_loops_no_ref[idx -1], closest_loop_dram]) != set(['R', 'S']):
                data_amount_refresh = prod_non_XY * Xi * (self.pred_in['l']['R'] -1) if closest_loop_dram == 'R' else prod_non_XY * (self.pred_in['l']['S'] -1) * Yi
                NR = True if closest_loop_dram == 'R' else False
                NS = True if closest_loop_dram == 'S' else False
            else:
                # v1 needs to be added with a scale and an offset to to compensate the actual changed refresh time (this not the actually amount for init load)
                data_amount_refresh = prod_non_XY * ( Xi * (self.pred_in['l']['R'] -1) + Yi* (self.pred_in['l']['S'] -1) + self.pred_in['l']['S'] * self.pred_in['l']['R'] )
                NSR = True
        elif closest_loop_sram in ['R', 'S']:
            idx = self.sram_loops_above_refresh_loc.index(closest_loop_sram)
            Xi_new = max(self.pred_in['l']['U'], (Xo_sram*Xo_noc*Xo_rf-1)*self.pred_in['l']['U'] )
            Yi_new = max(self.pred_in['l']['U'], (Yo_sram*Yo_noc*Yo_rf-1)*self.pred_in['l']['U'] )
            if (idx == 0 and set([self.dram_loops_no_ref[-1], closest_loop_sram]) != set(['R', 'S'])) or set([sram_loops_above_refresh_loc[idx -1], closest_loop_sram]) != set(['R', 'S']):
                # data_amount_refresh = prod_non_XY * Xi * Yi_new if closest_loop_dram == 'R' else prod_non_XY * Xi_new * Yi
                data_amount_refresh = prod_non_XY * Xi * (self.pred_in['l']['R'] -1) if closest_loop_sram == 'R' else prod_non_XY * (self.pred_in['l']['S'] -1) * Yi
                NR = True if closest_loop_sram == 'R' else False
                NS = True if closest_loop_sram == 'S' else False
            else:
                # v1 needs to be added with a scale and an offset to to compensate the actual changed refresh time (this not the actually amount for init load)
                # data_amount_refresh = prod_non_XY * (Xi_new * Yi + Xi* Yi_new + (self.pred_in['l']['U']) **2)
                data_amount_refresh = prod_non_XY * ( Xi * (self.pred_in['l']['R'] -1) + Yi * (self.pred_in['l']['S'] -1) + self.pred_in['l']['S'] * self.pred_in['l']['R'] )
                NSR = True
        else:
            data_amount_refresh = data_amount_init_load
        return data_amount_init_load, data_amount_refresh, NX, NY, NS, NR, NSR
    def __input_rf_refresh_amount(self, rf_loops_above_refresh_loc, rf_loops_below_refresh_loc, ignore_noc_X=True, ignore_noc_Y=True, forward = False):
        tilings = self.pred_in['df']['tiling']
        val_to_prod_rf = [ tilings['rf'][loop_name]
                           for loop_name in rf_loops_below_refresh_loc
                           if loop_name in self.input_loops and loop_name not in ['Xo', 'Yo', 'R', 'S'] ]
        Xo_rf = tilings['rf']['Xo'] if 'Xo' in rf_loops_below_refresh_loc else 1
        Yo_rf = tilings['rf']['Yo'] if 'Yo' in rf_loops_below_refresh_loc else 1
        Xo_noc = tilings['noc']['Xo']['val'] if 'Xo' in self.noc_loops_no_ref and not ignore_noc_X else 1
        Yo_noc = tilings['noc']['Yo']['val'] if 'Yo' in self.noc_loops_no_ref and not ignore_noc_Y else 1
        Xo = Xo_rf * Xo_noc
        Yo = Yo_rf * Yo_noc
        Xi = (Xo-1)*self.pred_in['l']['U'] + self.pred_in['l']['S'] if 'S' in rf_loops_below_refresh_loc or 'Xo' in rf_loops_below_refresh_loc or forward else 1
        Yi = (Yo-1)*self.pred_in['l']['U'] + self.pred_in['l']['R'] if 'R' in rf_loops_below_refresh_loc or 'Yo' in rf_loops_below_refresh_loc or forward else 1
        prod_non_XY = math.prod(val_to_prod_rf)
        data_amount_init_load = prod_non_XY * Xi * Yi 
        closest_loop_rf = self.__find_closest_related_above(loops=rf_loops_above_refresh_loc, related_loops= ['Xo', 'Yo', 'R', 'S'])
        closest_loop_sram = self.__find_closest_related_above(loops=self.sram_loops_no_ref, related_loops= ['Xo', 'Yo', 'R', 'S'])
        closest_loop_dram = self.__find_closest_related_above(loops=self.dram_loops_no_ref, related_loops = ['Xo', 'Yo', 'R', 'S'])
        NX, NY, NS, NR, NSR = (False, False, False, False, False)
        if not self.pred_in['df']['ref']['rf'] or (closest_loop_sram == '' and closest_loop_dram == '' and closest_loop_rf == ''):# minumum refresh not enabled or related loop not found
            data_amount_refresh = data_amount_init_load
        elif (closest_loop_rf=='' and closest_loop_sram=='' and closest_loop_dram=='Xo' ) or (closest_loop_rf == '' and closest_loop_sram == 'Xo') or closest_loop_rf == 'Xo':
            assert 'S' in rf_loops_below_refresh_loc, 'when calculating input rf refresh volume: Xo found above refresh loc but no S found below refresh loc'
            Xi_new = Xo *  self.pred_in['l']['U']
            data_amount_refresh = prod_non_XY * Xi_new * Yi
            NX = True
        elif (closest_loop_sram == '' and closest_loop_dram == 'Yo') or closest_loop_sram == 'Yo':
            assert 'R' in rf_loops_below_refresh_loc, 'when calculating input rf refresh volume: Yo found above refresh loc but no R found below refresh loc'
            Yi_new = Yo * self.pred_in['l']['U']
            data_amount_refresh = prod_non_XY * Xi* Yi_new
            NY =  True
        elif (closest_loop_rf == '' and closest_loop_sram == '' and closest_loop_dram in ['R', 'S']):
            idx = self.dram_loops_no_ref.index(closest_loop_dram)
            Xi_new = max(self.pred_in['l']['U'], (Xo-1)*self.pred_in['l']['U'] )
            Yi_new = max(self.pred_in['l']['U'], (Yo-1)*self.pred_in['l']['U'] )
            if idx == 0 or set([self.dram_loops_no_ref[idx -1], closest_loop_dram]) != set(['R', 'S']):
                data_amount_refresh = prod_non_XY * Xi * (self.pred_in['l']['R'] -1) if closest_loop_dram == 'R' else prod_non_XY * (self.pred_in['l']['S'] -1) * Yi
                NR = True if closest_loop_dram == 'R' else False
                NS = True if closest_loop_dram == 'S' else False
            else:
                data_amount_refresh = prod_non_XY * ( Xi * (self.pred_in['l']['R'] -1) + Yi* (self.pred_in['l']['S'] -1) + self.pred_in['l']['S'] * self.pred_in['l']['R'] )
                NSR = True
        elif (closest_loop_rf == '' and closest_loop_sram in ['R', 'S']):
            idx = self.sram_loops_no_ref.index(closest_loop_sram)
            Xi_new = max(self.pred_in['l']['U'], (Xo-1)*self.pred_in['l']['U'] )
            Yi_new = max(self.pred_in['l']['U'], (Yo-1)*self.pred_in['l']['U'] )
            if (idx == 0 and set([self.dram_loops_no_ref[-1], closest_loop_sram]) != set(['R', 'S']) ) or set([self.sram_loops_no_ref[idx -1], closest_loop_sram]) != set(['R', 'S']):
                data_amount_refresh = prod_non_XY * Xi * (self.pred_in['l']['R'] -1) if closest_loop_sram == 'R' else prod_non_XY * (self.pred_in['l']['S'] -1) * Yi
                NR = True if closest_loop_sram == 'R' else False
                NS = True if closest_loop_sram == 'S' else False
            else:
                data_amount_refresh = prod_non_XY * ( Xi * (self.pred_in['l']['R'] -1) + Yi* (self.pred_in['l']['S'] -1) + self.pred_in['l']['S'] * self.pred_in['l']['R'] )
                NSR = True
        elif closest_loop_rf in ['R', 'S']:
            idx = rf_loops_above_refresh_loc.index(closest_loop_rf)
            Xi_new = max(self.pred_in['l']['U'], (Xo-1)*self.pred_in['l']['U'] )
            Yi_new = max(self.pred_in['l']['U'], (Yo-1)*self.pred_in['l']['U'] )
            if (idx == 0 and set([self.sram_loops_no_ref[-1], closest_loop_rf]) != set(['R', 'S'])) or set([rf_loops_above_refresh_loc[idx -1], closest_loop_rf]) != set(['R', 'S']):
                data_amount_refresh = prod_non_XY * Xi * (self.pred_in['l']['R'] -1) if closest_loop_rf == 'R' else prod_non_XY * (self.pred_in['l']['S'] -1) * Yi
                NR = True if closest_loop_rf == 'R' else False
                NS = True if closest_loop_rf == 'S' else False
            else:
                data_amount_refresh = prod_non_XY * ( Xi * (self.pred_in['l']['R'] -1) + Yi * (self.pred_in['l']['S'] -1) + self.pred_in['l']['S'] * self.pred_in['l']['R'] )
                NSR = True
        else:
            data_amount_refresh = data_amount_init_load
        return data_amount_init_load, data_amount_refresh, NX, NY, NS, NR, NSR, Xo, Yo
    def __sram_refresh_amount(self, sram_loops_below_refresh_loc, related_loops):
        tilings = self.pred_in['df']['tiling']
        val_to_prod_sram = [ tilings['sram'][loop_name]
                             for loop_name in sram_loops_below_refresh_loc
                             if loop_name in related_loops ]
        val_to_prod_noc = [ tilings['noc'][loop_name]['val']
                            for loop_name in self.noc_loops_no_ref
                            if loop_name in related_loops ]
        val_to_prod_rf = [ tilings['rf'][loop_name]
                           for loop_name in self.rf_loops_no_ref
                           if loop_name in related_loops ]
        return math.prod(val_to_prod_sram + val_to_prod_noc + val_to_prod_rf)
    def __rf_refresh_amount(self, rf_loops_below_refresh_loc, related_loops):
        tilings = self.pred_in['df']['tiling']
        val_to_prod_rf = [ tilings['rf'][loop_name]
                           for loop_name in rf_loops_below_refresh_loc
                           if loop_name in related_loops and 'refresh' not in loop_name  ]
        # print (val_to_prod_rf)
        return math.prod(val_to_prod_rf)
    def __noc_get_prod(self, related_loops): # parallel factor for related loops and unrelated loops :
        tilings = self.pred_in['df']['tiling']
        val_to_prod_related = [ tilings['noc'][loop_name]['val']
                                for loop_name in self.noc_loops_no_ref
                                if loop_name in related_loops ]
        val_to_prod_not_related = [ tilings['noc'][loop_name]['val']
                                    for loop_name in self.noc_loops_no_ref
                                    if loop_name not in related_loops ]
        return math.prod(val_to_prod_related), math.prod(val_to_prod_not_related)        
    def __find_closest_related_above(self, loops, related_loops):
        closest_loop = ''
        for each in loops[::-1]:
            if each in related_loops:
                closest_loop = each
                break
        return closest_loop
    def __sram_refresh_times(self, sram_loops_above_refresh_loc, related_loops):
        dram_loops = self.dram_loops_no_ref
        dram_tilings = self.pred_in['df']['tiling']['dram']
        sram_tilings = self.pred_in['df']['tiling']['sram']
        # from the refresh location find closest weight loop
        closest_loop_sram = self.__find_closest_related_above(sram_loops_above_refresh_loc, related_loops)
        closest_loop_dram = self.__find_closest_related_above(dram_loops, related_loops)
        if closest_loop_sram == '' and closest_loop_dram == '':
            val_to_prod = []
        elif closest_loop_sram == '' and closest_loop_dram != '':
            dram_loops_sel = dram_loops[:dram_loops.index(closest_loop_dram)+1]
            val_to_prod = [ dram_tilings[loop_name] for loop_name in dram_loops_sel ]
        else: #elif closest_loop_sram != '':
            sram_loops_sel = sram_loops_above_refresh_loc[:sram_loops_above_refresh_loc.index(closest_loop_sram)+1]
            val_to_prod = [ sram_tilings[loop] for loop in sram_loops_sel ] \
                        + [ dram_tilings[loop] for loop in dram_loops ]
        return math.prod(val_to_prod)
    def __input_sram_refresh_times(self, sram_loops_above_refresh_loc, related_loops, NX, NY, NS, NR, NSR):
        dram_loops = self.dram_loops_no_ref
        dram_tilings = self.pred_in['df']['tiling']['dram']
        sram_tilings = self.pred_in['df']['tiling']['sram']
        # from the refresh location find closest weight loop
        closest_loop_sram = self.__find_closest_related_above(sram_loops_above_refresh_loc, related_loops)
        closest_loop_dram = self.__find_closest_related_above(dram_loops, related_loops)
        if closest_loop_sram == '' and closest_loop_dram == '':
            val_to_prod = []
        elif closest_loop_sram == '' and closest_loop_dram != '':
            dram_loops_sel = dram_loops[:dram_loops.index(closest_loop_dram)+1]
            val_to_prod = [ dram_tilings[loop_name] for loop_name in dram_loops_sel ]
        else: #elif closest_loop_sram != '':
            sram_loops_sel = sram_loops_above_refresh_loc[:sram_loops_above_refresh_loc.index(closest_loop_sram)+1]
            val_to_prod = [ sram_tilings[loop] for loop in sram_loops_sel ] \
                        + [ dram_tilings[loop] for loop in dram_loops ]
        if NX or NY or NS or NR:
            val_to_prod = val_to_prod[:-1] # remove the closest X, Y, R, S
        elif NSR:
            val_to_prod = val_to_prod[:-2] # remove the closest R & S
        else:
            pass
        return math.prod(val_to_prod)
    def __rf_refresh_times(self, rf_loops_above_refresh_loc, related_loops):
        dram_loops = self.dram_loops_no_ref
        sram_loops = self.sram_loops_no_ref
        dram_tilings = self.pred_in['df']['tiling']['dram']
        sram_tilings = self.pred_in['df']['tiling']['sram']
        rf_tilings = self.pred_in['df']['tiling']['rf']
        # from the refresh location find closest weight loop
        closest_loop_rf = self.__find_closest_related_above(rf_loops_above_refresh_loc, related_loops)
        closest_loop_sram = self.__find_closest_related_above(sram_loops, related_loops)
        closest_loop_dram = self.__find_closest_related_above(dram_loops, related_loops)
        if closest_loop_rf == '' and closest_loop_sram == '' and closest_loop_dram == '':
            val_to_prod = []
        elif closest_loop_rf == '' and closest_loop_sram == '' and closest_loop_dram != '':
            dram_loops_sel = dram_loops[:dram_loops.index(closest_loop_dram)+1]
            
            val_to_prod = [ dram_tilings[loop_name] for loop_name in dram_loops_sel ]
        elif closest_loop_rf == '' and closest_loop_sram != '':
            sram_loops_sel = sram_loops[:sram_loops.index(closest_loop_sram)+1]
            val_to_prod = [ sram_tilings[loop_name] for loop_name in sram_loops_sel ] \
                        + [ dram_tilings[loop] for loop in dram_loops ]
        else: #elif closest_loop_rf != '':
            rf_loops_sel = rf_loops_above_refresh_loc[:rf_loops_above_refresh_loc.index(closest_loop_rf)+1]
            val_to_prod = [ rf_tilings[loop] for loop in rf_loops_sel ] \
                        + [ sram_tilings[loop] for loop in sram_loops ] \
                        + [ dram_tilings[loop] for loop in dram_loops ]
        # print (val_to_prod)
        return math.prod(val_to_prod)
    def __input_rf_refresh_times(self, rf_loops_above_refresh_loc, related_loops, NX, NY, NS, NR, NSR):
        dram_loops = self.dram_loops_no_ref
        sram_loops = self.sram_loops_no_ref
        dram_tilings = self.pred_in['df']['tiling']['dram']
        sram_tilings = self.pred_in['df']['tiling']['sram']
        rf_tilings = self.pred_in['df']['tiling']['rf']
        # from the refresh location find closest weight loop
        closest_loop_rf = self.__find_closest_related_above(rf_loops_above_refresh_loc, related_loops)
        closest_loop_sram = self.__find_closest_related_above(sram_loops, related_loops)
        closest_loop_dram = self.__find_closest_related_above(dram_loops, related_loops)
        if closest_loop_rf == '' and closest_loop_sram == '' and closest_loop_dram == '':
            val_to_prod = []
        elif closest_loop_rf == '' and closest_loop_sram == '' and closest_loop_dram != '':
            dram_loops_sel = dram_loops[:dram_loops.index(closest_loop_dram)+1]
            val_to_prod = [ dram_tilings[loop_name] for loop_name in dram_loops_sel ]
        elif closest_loop_rf == '' and closest_loop_sram != '':
            sram_loops_sel = sram_loops[:sram_loops.index(closest_loop_sram)+1]
            val_to_prod = [ sram_tilings[loop_name] for loop_name in sram_loops_sel ] \
                        + [ dram_tilings[loop] for loop in dram_loops ]
        else: #elif closest_loop_rf != '':
            rf_loops_sel = rf_loops_above_refresh_loc[:rf_loops_above_refresh_loc.index(closest_loop_rf)+1]
            val_to_prod = [ rf_tilings[loop] for loop in rf_loops_sel ] \
                        + [ sram_tilings[loop] for loop in sram_loops ] \
                        + [ dram_tilings[loop] for loop in dram_loops ]
        if NX or NY or NS or NR:
            val_to_prod = val_to_prod[:-1] # remove the closest X, Y, R, S
        elif NSR:
            val_to_prod = val_to_prod[:-2] # remove the closest R & S
        else:
            pass
        # print (val_to_prod)
        return math.prod(val_to_prod)
    def __output_sram_refresh_times(self, sram_loops_above_refresh_loc):
        dram_loops = self.dram_loops_no_ref
        dram_tilings = self.pred_in['df']['tiling']['dram']
        sram_tilings = self.pred_in['df']['tiling']['sram']
        # from the refresh location find closest weight loop
        closest_loop_sram = self.__find_closest_related_above(sram_loops_above_refresh_loc, self.output_loops )
        closest_loop_dram = self.__find_closest_related_above(dram_loops, self.output_loops )
        if closest_loop_sram == '' and closest_loop_dram == '':
            val_to_prod = []
            C_val_to_prod = []
        elif closest_loop_sram == '' and closest_loop_dram != '':
            dram_loops_sel = dram_loops[:dram_loops.index(closest_loop_dram)+1]
            C_val_to_prod = [ dram_tilings[loop_name] for loop_name in dram_loops_sel if loop_name == 'C']
            val_to_prod = [ dram_tilings[loop_name] for loop_name in dram_loops_sel ]
        else: # elif closest_loop_sram != '':
            sram_loops_sel = sram_loops_above_refresh_loc[:sram_loops_above_refresh_loc.index(closest_loop_sram)+1]
            val_to_prod = [ sram_tilings[loop] for loop in sram_loops_sel ] \
                        + [ dram_tilings[loop] for loop in dram_loops ]
            C_val_to_prod = [ sram_tilings[loop] for loop in sram_loops_sel if loop == 'C'] \
                          + [ dram_tilings[loop] for loop in dram_loops  if loop == 'C']
        C = math.prod(C_val_to_prod)
        # print (val_to_prod)
        act_refresh_times = int( math.prod(val_to_prod) / C)
        psum_refresh_times = act_refresh_times * (2*C-1)
        return act_refresh_times, psum_refresh_times, C
    def __output_rf_refresh_times(self, rf_loops_above_refresh_loc):
        dram_loops = self.dram_loops_no_ref
        sram_loops = self.sram_loops_no_ref
        dram_tilings = self.pred_in['df']['tiling']['dram']
        sram_tilings = self.pred_in['df']['tiling']['sram']
        rf_tilings = self.pred_in['df']['tiling']['rf']
        # from the refresh location find closest weight loop
        closest_loop_rf = self.__find_closest_related_above(rf_loops_above_refresh_loc, self.output_loops )
        closest_loop_sram = self.__find_closest_related_above(sram_loops, self.output_loops )
        closest_loop_dram = self.__find_closest_related_above(dram_loops, self.output_loops )
        if closest_loop_rf == '' and closest_loop_sram == '' and closest_loop_dram == '':
            val_to_prod = []
            C_val_to_prod = []
        elif closest_loop_rf == '' and closest_loop_sram == '' and closest_loop_dram != '':
            dram_loops_sel = dram_loops[:dram_loops.index(closest_loop_dram)+1]
            C_val_to_prod = [ dram_tilings[loop_name] for loop_name in dram_loops_sel if loop_name == 'C']
            val_to_prod = [ dram_tilings[loop_name] for loop_name in dram_loops_sel ]
        elif closest_loop_rf == '' and closest_loop_sram != '':
            sram_loops_sel = sram_loops[:sram_loops.index(closest_loop_sram)+1]    
            val_to_prod = [ sram_tilings[loop] for loop in sram_loops_sel ] \
                        + [ dram_tilings[loop] for loop in dram_loops  ]
            C_val_to_prod = [ sram_tilings[loop] for loop in sram_loops_sel if loop == 'C'] \
                          + [ dram_tilings[loop] for loop in dram_loops  if loop == 'C']
        else: # elif closest_loop_rf != ''
            rf_loops_sel = rf_loops_above_refresh_loc[:rf_loops_above_refresh_loc.index(closest_loop_rf)+1]
            val_to_prod = [ sram_tilings[loop] for loop in sram_loops ] \
                        + [ dram_tilings[loop] for loop in dram_loops ] \
                        + [ rf_tilings[loop] for loop in rf_loops_sel ]
            C_val_to_prod = [ sram_tilings[loop] for loop in sram_loops if loop == 'C'] \
                          + [ dram_tilings[loop] for loop in dram_loops  if loop == 'C'] \
                          + [ rf_tilings[loop] for loop in rf_loops_sel if loop == 'C']
        C = math.prod(C_val_to_prod)
        act_refresh_times = int( math.prod(val_to_prod) / C)
        psum_refresh_times = act_refresh_times * (2*C-1)
        return act_refresh_times, psum_refresh_times, C
    def calc_weight_dram_access(self): # dram here means dram read and sram write (byte)
        sram_loops_below, sram_loops_above = self.__split_loops_by_refresh(self.sram_loops, 'W_refresh')
        data_amount = self.__sram_refresh_amount(sram_loops_below_refresh_loc=sram_loops_below,
                                                 related_loops = self.weight_loops)
        times = self.__sram_refresh_times(sram_loops_above_refresh_loc=sram_loops_above,
                                          related_loops=self.weight_loops)
        self.__check_buf_size_singly('sram','W', data_amount)
        self.access['dram']['W'] = times * data_amount *self.pred_in['l']['prec']['W']
    def calc_output_dram_access(self): # dram here means dram read & write and sram read & write (byte)
        sram_loops_below, sram_loops_above = self.__split_loops_by_refresh(self.sram_loops, 'O_refresh')
        data_amount = self.__sram_refresh_amount(sram_loops_below_refresh_loc=sram_loops_below,
                                                 related_loops = self.output_loops)
        self.__check_buf_size_singly('sram','O', data_amount)
        act_ref_times, psum_ref_times, C = \
            self.__output_sram_refresh_times(sram_loops_above_refresh_loc=sram_loops_above)
        self.access['dram']['O'] = ( act_ref_times * self.pred_in['l']['prec']['A'] 
                                   + psum_ref_times * self.pred_in['l']['prec']['P'] ) * data_amount  \
                                   if C > 1 else act_ref_times * self.pred_in['l']['prec']['A'] * data_amount
    def calc_weight_sram_rf_noc_access(self):
        multicast = self.pred_in['hw']['noc']['multicast']['W']
        rf_loops_below, rf_loops_above = self.__split_loops_by_refresh(self.rf_loops, 'W_refresh')
        # volume V: same as dram
        data_amount = self.__rf_refresh_amount(rf_loops_below_refresh_loc=rf_loops_below,
                                               related_loops = self.weight_loops)
        self.__check_buf_size_singly('rf','W', data_amount)
        # times N: above loops ignore noc loops (dram/sram/rf loops)
        times = self.__rf_refresh_times(rf_loops_above_refresh_loc=rf_loops_above,
                                        related_loops= self.weight_loops)
        # times P: multicast if false, prod all; otherwise mutiply related
        Pr, Pnr = self.__noc_get_prod(related_loops = self.weight_loops) # parallel factor for related loops and unrelated loops 
        self.access['sram']['W'] = times * data_amount * Pr * self.pred_in['l']['prec']['W'] if multicast else times * data_amount * Pr * Pnr * self.pred_in['l']['prec']['W']
        self.access['noc_forward']['W'] = 0
        self.access['noc_unicast']['W'] = 0 if multicast else times * data_amount * Pr * Pnr * self.pred_in['l']['prec']['W']
        self.access['noc_multicast']['W'] = times * data_amount * Pr * self.pred_in['l']['prec']['W'] if multicast else 0
        self.access['rf']['W'] = times * data_amount * Pr * Pnr * self.pred_in['l']['prec']['W']
    def calc_output_sram_rf_noc_access(self):
        forward = self.pred_in['hw']['noc']['forward']['O']
        rf_loops_below, rf_loops_above = self.__split_loops_by_refresh(self.rf_loops, 'O_refresh')
        # volume: same
        data_amount = self.__rf_refresh_amount(rf_loops_below_refresh_loc=rf_loops_below,
                                               related_loops = self.output_loops)
        self.__check_buf_size_singly('rf','O', data_amount)
        act_ref_times, psum_ref_times, C = self.__output_rf_refresh_times(rf_loops_above_refresh_loc=rf_loops_above)
        assert C >= 1, 'Pnr must be larger than 1, but got {val} instead'.format(C)
        Pr, Pnr = self.__noc_get_prod(related_loops = self.output_loops) # parallel factor for related loops and unrelated loops 
        assert Pnr >= 1, 'Pnr must be larger than 1, but got {val} instead'.format(Pnr)
        assert Pnr == 1 or forward, 'no partial sum forward which wastes energy'
        self.access['sram']['O'] = ( psum_ref_times * Pr * self.pred_in['l']['prec']['P'] ) * data_amount \
                                   if C > 1 else act_ref_times * Pr * self.pred_in['l']['prec']['A'] * data_amount
        Nnc = act_ref_times
        Nc = Nnc * C
        self.access['rf']['O'] = Nnc * Pr * ( 2*C*Pnr - 1)  * data_amount * self.pred_in['l']['prec']['P'] if (C > 1 or Pnr > 1) \
                            else Nnc * Pr * data_amount * self.pred_in['l']['prec']['A']
        self.access['noc_unicast']['O'] = Nc * Pr * data_amount * self.pred_in['l']['prec']['P'] if C > 1 \
                                     else Nnc * Pr * data_amount * self.pred_in['l']['prec']['A']
        self.access['noc_forward']['O'] = 0 if Pnr==1 else Nc * Pr * (Pnr -1) * data_amount * self.pred_in['l']['prec']['P']
    def calc_input_sram_rf_noc_access(self):
        forward = self.pred_in['hw']['noc']['forward']['I']
        min_refresh = self.pred_in['df']['ref']['rf']
        multicast = self.pred_in['hw']['noc']['multicast']['I']
        rf_loops_below, rf_loops_above = self.__split_loops_by_refresh(self.rf_loops, 'I_refresh')
        Pr, Pnr = self.__noc_get_prod(related_loops = self.input_loops) # parallel factor for related loops and unrelated loops 
        V1, V2, NX, NY, NS, NR, NSR, _, _ = self.__input_rf_refresh_amount(rf_loops_above_refresh_loc=rf_loops_above, rf_loops_below_refresh_loc=rf_loops_below)
        self.__check_buf_size_singly('rf','I', V1)
        N = self.__input_rf_refresh_times(rf_loops_above_refresh_loc=rf_loops_above,
                                          related_loops=self.input_loops,
                                          NX = NX, NY = NY, NS = NS, NR = NR, NSR = NSR)
        noc_related_loops = [loop_name for loop_name in self.noc_loops_no_ref if loop_name in self.input_loops]
        if (not forward and not min_refresh) or (not forward and min_refresh and not NX and not NY and not NS and not NR and not NSR):
            self.access['rf']['I'] = N * Pr * Pnr * V1 * self.pred_in['l']['prec']['A']
            self.access['sram']['I'] = N * Pr * V1 * self.pred_in['l']['prec']['A'] if multicast else N * Pr * Pnr * V1 * self.pred_in['l']['prec']['A']
            self.access['noc_unicast']['I'] = 0 if multicast else N * Pr * Pnr * V1 * self.pred_in['l']['prec']['A']
            self.access['noc_multicast']['I'] = N * Pr * V1 * self.pred_in['l']['prec']['A'] if multicast else 0
        elif not forward and min_refresh and (NX or NY or NS or NR or NSR):
            res = N * (V1 + V2) * Pr * self.pred_in['l']['prec']['A']
            self.access['sram']['I'] = res if multicast else res * Pnr
            self.access['noc_multicast']['I'] = res if multicast else 0
            self.access['noc_unicast']['I'] = 0 if multicast else res * Pnr
            self.access['rf']['I'] = res * Pnr
        elif forward:
            assert 'Xo' in noc_related_loops or 'Yo' in noc_related_loops, 'Pr must include either Yo or Xo, but got [{noc_loops}] instead'.format(noc_loops=str(noc_related_loops))
            if min_refresh and NS and 'Xo' in noc_related_loops and 'Yo' not in noc_related_loops:
                N = N /self.pred_in['l']['S']
                V1_prime, V2_prime, _, _, _, _, _, _, _ = self.__input_rf_refresh_amount(rf_loops_above_refresh_loc=rf_loops_above, rf_loops_below_refresh_loc=rf_loops_below, ignore_noc_X=False, ignore_noc_Y=True, forward=True)
                Pr_prime = Pr / self.pred_in['df']['tiling']['noc']['Xo']['val']
                res = N * (V1_prime + V2_prime ) * Pr_prime * self.pred_in['l']['prec']['A']
                self.access['sram']['I'] = res if multicast else res * Pnr
                self.access['noc_unicast']['I'] = 0 if multicast else res * Pnr
                self.access['noc_multicast']['I'] = res if multicast else 0
                self.access['noc_forward']['I'] = V2 * N * Pr * Pnr
                self.access['rf']['I'] = res * Pnr

            elif min_refresh and NR and 'Yo' in noc_related_loops and 'Xo' not in noc_related_loops:
                N = N /self.pred_in['l']['R']
                V1_prime, V2_prime, _, _, _, _, _, _, _ = self.__input_rf_refresh_amount(rf_loops_above_refresh_loc=rf_loops_above, rf_loops_below_refresh_loc=rf_loops_below, ignore_noc_X=True, ignore_noc_Y=False, forward=True)
                Pr_prime = Pr / self.pred_in['df']['tiling']['noc']['Yo']['val']
                res = N * (V1_prime + V2_prime) * Pr_prime * self.pred_in['l']['prec']['A']
                self.access['sram']['I'] = res if multicast else res * Pnr
                self.access['noc_unicast']['I'] = 0 if multicast else res * Pnr
                self.access['noc_multicast']['I'] = res if multicast else 0
                self.access['noc_forward']['I'] = V2 * N * Pr * Pnr
                self.access['rf']['I'] = res * Pnr
            elif min_refresh and NSR and 'Xo' in noc_related_loops and 'Yo' in noc_related_loops:
                N = N / (self.pred_in['l']['S'] * self.pred_in['l']['S']) 
                V1_prime, V2_prime, _, _, _, _, _, Xo, Yo = self.__input_rf_refresh_amount(rf_loops_above_refresh_loc=rf_loops_above, rf_loops_below_refresh_loc=rf_loops_below, ignore_noc_X=False, ignore_noc_Y=False, forward = True)
                Pr_prime = Pr / self.pred_in['df']['tiling']['noc']['Xo']['val'] / self.pred_in['df']['tiling']['noc']['Yo']['val']
                res = N * (V1_prime + V2_prime) * Pr_prime * self.pred_in['l']['prec']['A']
                self.access['sram']['I'] = res if multicast else res * Pnr
                self.access['noc_unicast']['I'] = 0 if multicast else res * Pnr
                self.access['noc_multicast']['I'] = res if multicast else 0
                self.access['noc_forward']['I'] = V2 * N * Pr_prime * (Xo + Yo) * Pnr * self.pred_in['l']['prec']['A']
                self.access['rf']['I'] = res * Pnr
            else:
                print (NS)
                assert False, 'this case is not supported'
        else:
            assert False, 'this case is not supported'
    def calc_input_dram_access(self):# dram here means dram read and sram write (byte)
        min_refresh = self.pred_in['df']['ref']['rf']
        sram_loops_below, sram_loops_above = self.__split_loops_by_refresh(self.sram_loops, 'I_refresh') 
        V1, V2, NX, NY, NS, NR, NSR = self.__input_sram_refresh_amount(sram_loops_above_refresh_loc=sram_loops_above, sram_loops_below_refresh_loc=sram_loops_below)
        self.__check_buf_size_singly('sram','I', V1)
        N = self.__input_sram_refresh_times(sram_loops_above_refresh_loc=sram_loops_above,
                                            related_loops=self.input_loops,
                                            NX = NX, NY = NY, NS = NS, NR = NR, NSR = NSR)
        if ( not min_refresh) or (min_refresh and not NX and not NY and not NS and not NR and not NSR):
            self.access['dram']['I'] = N * V1 * self.pred_in['l']['prec']['A']
        elif min_refresh and (NX or NY or NS or NR or NSR):
            self.access['dram']['I'] = N * (V1+V2) * self.pred_in['l']['prec']['A']
        else:
            assert False, 'this case is not supported'
    def calc_comp(self):
        l = self.pred_in['l']
        self.comp = l['N'] * l['C'] * l['Xo'] * l['Yo'] * l['K'] * l['R'] * l['S']
    def calc_energy(self):
        for data in ['I', 'O', 'W']:
            self.energy['dram'][data] = self.access['dram'][data] * self.pred_in['hw']['dram']['energy']
            self.energy['sram'][data] = self.access['sram'][data] * self.pred_in['hw']['sram'] [ self.pred_in['hw']['sram']['alloc'][data] ]['energy']
            self.energy['rf'][data] = self.access['rf'][data] * self.pred_in['hw']['rf'] [ self.pred_in['hw']['rf']['alloc'][data] ]['energy']
            self.energy['comp'][data] = self.comp * self.pred_in['hw']['pe']['energy']/3.0
            self.energy['noc'][data] = sum([ self.access['noc_'+noc_type][data] * self.pred_in['hw']['noc'][noc_type]['energy'] for noc_type in ['unicast', 'multicast', 'forward'] if self.pred_in['hw']['noc'][noc_type][data] ])
    def calc_latency(self):
        for data in ['I', 'O', 'W']:
            self.latency['sram'][data] = self.access['sram'][data] / self.pred_in['hw']['sram'] [ self.pred_in['hw']['sram']['alloc'][data] ]['bw']
            self.latency['comp'][data] = self.comp / (self.active_pe_rows * self.active_pe_cols)
            self.latency['noc'][data] = max([ self.access['noc_'+noc_type][data] / self.pred_in['hw']['noc'][noc_type]['bw'] for noc_type in ['unicast', 'multicast', 'forward'] if self.pred_in['hw']['noc'][noc_type][data] ])

    def run(self):
        success = True
        try:
            self.__check_loop_bounds()
            self.__check_RS_loops()
            self.__check_pe_util()
            self.calc_weight_dram_access()
            self.calc_output_dram_access()
            self.calc_input_dram_access()
            self.calc_weight_sram_rf_noc_access()
            self.calc_output_sram_rf_noc_access()
            self.calc_input_sram_rf_noc_access()
            self.__check_buf_size_all()
            self.calc_comp()
            self.calc_energy()
            self.calc_latency()
        except Exception as e:
            success = False
            self.calc_input_sram_rf_noc_access()
            print ('got the error message of [{err}]'.format(err = str(e)))
        return success
    def __save_csv(self, sv_name, dic2d, topic):
        with open(sv_name, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([topic,'input', 'output', 'weight'])
            for each_row in dic2d:
                writer.writerow([each_row, dic2d[each_row]['I'], dic2d[each_row]['O'], dic2d[each_row]['W']])
    def save(self, sv_name):
        self.__save_csv(sv_name+'_energy.csv', self.energy, 'energy (pJ)')
        self.__save_csv(sv_name+'_latency.csv', self.latency, 'latency (cycle)')
        self.__save_csv(sv_name+'_access.csv', self.access, 'access count (byte)')
    def __get_single_item_or_proc(self, dic_2d, keys, operation):
        keys = [keys] if isinstance(keys, str) else keys
        list_to_proc = [dic_2d[i][j] for i in dic_2d for j in dic_2d[i] if set(keys).issubset( set([i,j]) )]
        return operation(list_to_proc)
    def get_energy(self, keys=[]):
        return self.__get_single_item_or_proc(self.energy, keys, sum)
    def get_access_count(self, keys=[]):
        return self.__get_single_item_or_proc(self.access, keys, sum)
    def get_latency(self, keys=[]):
        return self.__get_single_item_or_proc(self.latency, keys, max)
    def get_buffer_size(self, key1, key2):
        return self.pred_in['hw'][key1][ self.pred_in['hw'][key1]['alloc'][key2] ]['size']



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chip Predictor')
    parser.add_argument('-l', '--layer', default='', type=str, help='path of YAML file for conv layer information')
    parser.add_argument('--hw', '--hardware', default='', type=str, help='path of YAML file for hardware configuration')
    parser.add_argument('--df', '--dataflow', default='', type=str, help='path of YAML file for dataflow configuration')
    parser.add_argument('-o', '--out-name', default='', type=str, help='output csv file name')
    args = parser.parse_args()
    predictor = convPrediction(args.hw, args.df, args.layer)
    res = predictor.run()
    if res: # success
        print ( 'Total energy (pJ): {val}'.format(val = predictor.get_energy() ) )
        print ( 'Overall latency (cycle): {val}'.format(val = predictor.get_latency() ) )
        print ( 'Throughput (frame/cycle): {val}'.format(val = round(predictor.pred_in['l']['N'] / predictor.get_latency(),10) ))
        predictor.save(args.out_name)
