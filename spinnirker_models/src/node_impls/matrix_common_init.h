#include <stdint.h>

typedef struct {
    //! The width of the matrix of data
    uint32_t width;
    //! The height of the matrix of data
    uint32_t height;
    //! The data values (width * height in size)
    int32_t data[];
} matrix_config_t;

typedef struct {
    //! The width of the matrix of data
    uint32_t width;
    //! The height of the matrix of data
    uint32_t height;
    //! The index of the component for DMA identification
    uint32_t component_index : 30;
    //! Whether the data is in SDRAM or not
    uint32_t in_sdram : 1;
    //! Whether a DMA is in progress or not
    uint32_t dma_in_progress : 1;
    //! A space to read data into, but only if in SDRAM
    int32_t *local_data[2];
    //! The index of local data to read from
    uint32_t read_index;
    //! The index of local data to write to
    uint32_t write_index;
    //! The matrtix data values; might be in SDRAM if not sufficient space
    int32_t *data;
} matrix_data_t;

extern matrix_data_t *matrix_init(uint32_t index, matrix_config_t *config,
        matrix_data_t *data);
