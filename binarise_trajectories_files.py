from os import system
from numpy import int64, int16, float64, float16
from pandas import read_table, read_pickle


def x64tox16(dtype):
    if dtype == int64: return int16
    if dtype == float64:
        return float16
    else:
        raise ValueError


def binarise(time_slots_):

    for time_slot in time_slots_:
        print(f' > Load time slot: {time_slot}')
        src_file_name = f'traffic-data/xy-trajectories/{time_slot}.txt'
        df = read_table(src_file_name, sep='\s+', header=None, names=(
            'Vehicle ID',
            'Frame ID',
            'Total Frames',
            'Global Time',
            'Local X',
            'Local Y',
            'Global X',
            'Global Y',
            'Vehicle Length',
            'Vehicle Width',
            'Vehicle Class',
            'Vehicle Velocity',
            'Vehicle Acceleration',
            'Lane Identification',
            'Preceding Vehicle',
            'Following Vehicle',
            'Spacing',
            'Headway'
        ))

        print(' > Drop unnecessary fields')
        df.drop(columns=[
            'Total Frames',
            'Global Time',
            'Global X',
            'Global Y',
            'Vehicle Class',
            'Vehicle Acceleration',
            'Preceding Vehicle',
            'Following Vehicle',
            'Spacing',
            'Headway',
        ], inplace=True)

        print(' > Cast {int,float}64 to {int,float}16, from 16 to 4 bytes per value')
        print('   Source data frame data types:', df.dtypes, sep='\n')
        src_columns_dtype = dict(df.dtypes)
        dst_columns_dtype = {k: x64tox16(v) for k, v in src_columns_dtype.items()}
        df = df.astype(dtype=dst_columns_dtype)
        print('   Destination data frame data types:', df.dtypes, sep='\n')

        print(' > Save binary (pickled) file')
        dst_file_name = f'traffic-data/xy-trajectories/{time_slot}.pkl'
        df.to_pickle(dst_file_name)

        print(' > Source and destination files')
        system(f'ls -lh {src_file_name}')
        system(f'ls -lh {dst_file_name}')


if __name__ == '__main__':

    time_slots = (
        'i80/trajectories-0400-0415',
        'i80/trajectories-0500-0515',
        'i80/trajectories-0515-0530',
    )
    binarise(time_slots)