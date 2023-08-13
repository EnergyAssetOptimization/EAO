import numpy as np
import pandas as pd
import datetime as dt
import copy

from eaopack.portfolio import Portfolio
from eaopack.optimization import Results, OptimProblem
from eaopack import serialization 


def extract_output(portf: Portfolio, op: OptimProblem, res:Results, prices: dict = None) -> dict:
    """ extract results into dataframes, containing readable 
        results such as asset dispatch and dcf

    Args:
        portf (Portfolio): portfolio optimized
        op (OptimProblem): optimization problem 
        res (Results): result
        prices (dict): used prices. Defaults to None if not to be added to output

    Returns: dictionary with results
        disp:    dataframe with dispatch per asset
        internal_variables: dataframe with internal variables per asset
        dcfs:    dataframe with discounted cash flows per asset
        prices:  dataframe with nodal prices and given prices
        special: dataframe with specific asset parameters (such as size in scaled assets)
    """
    output = {}
    output['summary'] = {}
    if isinstance(res, str): # not successful
        output['summary']['status']  = res
        output['DCF']       = None
        output['dispatch']  = None
        output['internal_variables']  = None
        output['prices']    = None        
        output['special']   = None
    else:
        output['summary']['status'] = 'successful'
        output['summary']['value']  = res.value
        output['summary'] = pd.DataFrame.from_dict(output['summary'], orient = 'index')
        output['summary'].index.name = 'Parameter'
        output['summary'].rename(columns={0: 'Values'}, inplace=True)

        # collecting result vectors in dataframes
        times   = portf.timegrid.timepoints # using time as main index
        dcfs    = pd.DataFrame(index = times) # dcf
        disp    = pd.DataFrame(index = times) # dispatch
        internal_variables = pd.DataFrame(index = times)
        duals   = pd.DataFrame(index = times) # duals of problem -  correspond to nodal prices
        special = pd.DataFrame(columns=['asset', 'variable', 'value', 'costs']) # one line per asset / parameter
        # extract dcf for all assets and set together
        for a in portf.assets:
            dcfs[a.name] = a.dcf(optim_problem = op, results = res)
            I =   (op.mapping['asset'] == a.name) \
                & (op.mapping['type'] == 'd')       # type dispatch
            my_mapping = op.mapping.loc[I,:]
        # extract dispatch per asset and node
        # in case an asset links nodes, dispatch should be separate per node
        for a in portf.assets:
            for i_n, n in enumerate(a.nodes):
                I =   (op.mapping['asset'] == a.name) \
                    & (op.mapping['type'] == 'd')      \
                    & (op.mapping['node'] == n.name)
                my_mapping = op.mapping.loc[I,:]
                if len(portf.nodes)==1:
                    myCol = a.name
                else: # add node information
                    myCol = (a.name +' ('+  n.name + ')')
                disp[myCol] = 0.
                for i,r in my_mapping.iterrows():
                    disp.loc[times[r.time_step], myCol] += res.x[i]*r.disp_factor
        # extract internal variables per asset
        for a in portf.assets:
            variable_names = op.mapping[(op.mapping['asset'] == a.name)& (op.mapping['type'] == 'i')]['var_name'].unique()
            for v in variable_names:
                I =   (op.mapping['asset'] == a.name) \
                    & (op.mapping['type'] == 'i')      \
                    & (op.mapping['var_name'] == v)
                my_mapping = op.mapping.loc[I,:]
                myCol = (a.name +' ('+  v + ')')
                internal_variables[myCol] = None
                for i,r in my_mapping.iterrows():
                    pass
                    internal_variables.loc[times[r.time_step], myCol] = res.x[i]
        # extract duals from nodal restrictions
        # looping through nodes and their recorded nodal restrictions and extract dual
        if not res.duals is None and not res.duals['N'] is None:
            ### new version withour nodal_restr column in mapping, rather explicit reference to time and node in OP
            for ii, id in enumerate(op.map_nodal_restr):
                name_nodal_price = 'nodal price: '+ id[1]
                duals.loc[times[id[0]], name_nodal_price] = -res.duals['N'][ii]
                pass
            pass
        else:   
            duals = pd.DataFrame()
        # extract specific parameters for all assets
        # specific parameters are named in the mapping (not 'd', 'i' ,... )
        not_special = ['d', 'i']
        for a in portf.assets:
            I =       (op.mapping['asset'] == a.name) \
                    & (~op.mapping['type'].isin(not_special))
            for i, r in op.mapping[I].iterrows():
                myrow = {'asset'      : r.asset,
                         'variable'   : r.type,
                         'value'      : res.x[i],
                         'costs'      : res.x[i]*op.c[i]
                           }
                special = special.append(myrow, ignore_index = True)
        # add given prices to duals in output (relevant reference)
        if not prices is None:
            for myc in prices:
                duals['input data: '+myc] = prices[myc]

        # In case the result comes from an SLP, we cannot sum up the dispatch across samples.
        # rather it should be the average. Therefore divide summed dispatch by number of samples
        # (only for future values of SLP)
        SLPcols = [col for col in op.mapping.columns if 'slp_step' in col]
        for myc in SLPcols:
            is_sample = op.mapping[myc].notnull()
            n_samples = len(op.mapping.loc[is_sample,myc].unique())
            I_t       = op.mapping.loc[is_sample,'time_step']
            disp.loc[times[I_t],:] = disp.loc[times[I_t],:]/n_samples
        output['DCF']       = dcfs
        output['dispatch']  = disp
        output['internal_variables'] = internal_variables
        output['prices']    = duals
        output['special']   = special

    return output

def output_to_file(output, file_name:str, format_output:str = 'xlsx',csv_ger:bool = False):
    """ write extracted output to file(s)

    Args:
        output ([type]): Target file (excel)
        file_name (str): file name
        format_output (str)    : xlsx, csv. format of output file. Defaults to 'xlsx'
        csv_ger (bool)      : English (False) or German (True) csv format. Defaults to False.            
    """
    for myk in output:
        if not isinstance(output[myk], pd.DataFrame):
            if output[myk] is None:
                output[myk] = pd.DataFrame() 
            elif isinstance(output[myk], dict):
                output[myk] = pd.DataFrame.from_dict(output[myk], orient = 'index')
    if (format_output.lower() == 'xlsx') or (format_output.lower() == 'xls'):
        writer = pd.ExcelWriter(file_name)
        for myk in output:
            if isinstance(output[myk].index, pd.DatetimeIndex):
                if not output[myk].index.tzinfo is None:
                    output[myk].index = output[myk].index.tz_localize(None)
            output[myk].to_excel(writer, sheet_name = myk)
        writer.close()
    elif (format_output.lower() == 'csv'):
        if not csv_ger:
            sep = ','
            decimal = '.'
        else:
            sep = ';'
            decimal = ','
        for myk in output:
            output[myk].to_csv(myk+'_'+file_name, sep = sep, decimal = decimal)                
    else:
        raise NotImplementedError('output format - '+format_output+' - not implemented')

#### easy access to object parameters e.g. for assets & portfolio
## get tree, get parameter, set parameter
def get_params_tree(obj) -> (list, dict):
    """ get parameters of object - typically asset or portfolio

    Args:
        obj (object): object to analyze
    Returns 
        dict of parameters (nested)
        list of parameter names (nested)
    """

    def make_dict(dd) -> (list, dict):
        if isinstance(dd, list): dd_keys = range(0,len(dd))
        elif isinstance(dd, dict): dd_keys = list(dd)
        else: 
            return None, None
        keys = [] # record keys
        for k in dd_keys:
            if isinstance(dd[k], dict) or isinstance(dd[k], list): 
                if isinstance(dd[k], list): ks = range(0,len(dd[k]))
                if isinstance(dd[k], dict): ks = list(dd[k])
                for myk in ks:
                    if not isinstance(myk, list): l_myk = [myk]
                    else: l_myk = myk
                    if isinstance(dd[k][myk], dict) or isinstance(dd[k][myk], list):
                        _ , tk = make_dict(dd[k][myk])
                        for ttk in tk:
                            if not isinstance(ttk, list): l_ttk = [ttk]
                            else: l_ttk = ttk
                            keys.append([k] + l_myk + l_ttk)
                    else: keys.append([k] + l_myk)
            else: keys.append(k)

        return copy.deepcopy(dd), keys
        
    # serialize and back to get rid of superfluous parameters (not set during __init__)
    s = serialization.to_json(obj) # serialize using all specific definitions
    dd = serialization.json.loads(s) # create simple dict
    o, k = make_dict(dd)
    return k, o

def get_param(obj, path):
    """ get specific parameter from an object using the path as generated from get_params_tree

    Args:
        path (_type_): list down to parameter as given by get_params_tree (e.g. [asset, timegrid, start] --> value)
    """
    def get(d,l):
        """ recursion to access nested dict """
        if len(l) == 1: return d[l[0]]
        return get(d[l[0]], l[1:])
    
    k,o = get_params_tree(obj)
    if not isinstance(path, list): path = [path]
    return get(o, path)
    

def set_param(obj, path, value):
    """ Set parameters of EAO objects. Limited checks, but facilitating managing nested objects such as portfolios or assets

    Args:
        obj (object): Object to manipulate

    Returns:
        obj (object): Manipulated object
        status (bool): True - > successful, False -> not successful
    """
    def sett(o,l,v):
        """ recursion to access nested object """
        if len(l) == 1: # leaf in tree. Set value
            o[l[0]] = v
            return o
        return sett(o[l[0]], l[1:], v) # recursion. Go down path
    
    if not isinstance(path, list): path = [path]
    k,o = get_params_tree(obj)
    _ = sett(o, path, value) # manipulates dict. output not needed
    
    try:
        s = serialization.json.dumps(o)
        res = serialization.load_from_json(s) # create object again (properly initializing)
    except:
        if 'name' in o: n = o['name']
        else: n = 'NA'
        raise ValueError('Error. Object could not be created. Parameter issue? Object: '+n+' | parameter '+str(path))
    return res

