from configs import bin_width


def get_bin_by_mz(mz):
    return int(mz / bin_width)

def get_mz_by_bin(bin_id):
    return (bin_id + 0.5) * bin_width