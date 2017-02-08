import pytest
from io_utils import OFFMatrix


def test_get_off_file():
    off_matrix = OFFMatrix(datapath="./test/", dataset="bathtub_0001.off")
    print(off_matrix.data)
    assert off_matrix.file_format == "OFF"
    assert off_matrix.num_vertices == 3514
    assert off_matrix.num_faces == 3546
    assert off_matrix.num_edges == 0

if __name__ == "__main__":
    pytest.main([__file__])