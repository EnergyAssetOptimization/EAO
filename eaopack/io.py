import numpy as np
import pandas as pd
import datetime as dt

from eaopack.portfolio import Portfolio
from eaopack.optimization import Results, OptimProblem


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
                    disp.loc[times[r.time_step], myCol] += res.x[i]
        # extract duals from nodal restrictions
        # looping through nodes and their recorded nodal restrictions and extract dual
        for i_node, n in enumerate(portf.nodes):
            name_nodal_price = 'nodal price: '+n
            duals[name_nodal_price] = np.nan # initialize column
            my_mapping = op.mapping.loc[op.mapping['node']==n,:]
            all_nr = my_mapping.loc[~my_mapping['nodal_restr'].isnull(),'nodal_restr'].unique()
            for nr in all_nr:
                # relevant info same for all; get first row that fits
                r = my_mapping[my_mapping['nodal_restr'] == nr].iloc[0]
                # attention: change sign to obtain nodal price from dual
                duals.loc[times[r.time_step], name_nodal_price] = -res.duals['N'][r['nodal_restr']]

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
                duals['input price: '+myc] = prices[myc]

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