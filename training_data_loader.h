#ifndef _DATA_LOADER_H_
#define _DATA_LOADER_H_
#include "activations/activations.h"

typedef struct Data
{
    uint64_t occupied; // occupancy without kings, A1 = 0
    int16_t eval;     
    int8_t result;     // -1 , 0 , 1
    uint8_t side;      // 1 for black 0 for white
    uint8_t wking;     // king square
    uint8_t bking;
    uint8_t packed[15]; // 2 pieces in one byte
} Data;

#define Mirror(sq) (sq ^ 56)
#define HorizontalMirror(sq) (sq ^ 7)

enum Pieces{
    PAWN   = 0, 
    KNIGHT = 1, 
    BISHOP = 2, 
    ROOK   = 3,  
    QUEEN  = 4, 
    KING   = 5,
    BLACK_PAWN   = 6, 
    BLACK_KNIGHT = 7,
    BLACK_BISHOP = 8, 
    BLACK_ROOK   = 9, 
    BLACK_QUEEN  = 10,  
    BLACK_KING   = 11,
};

enum Colors{
    WHITE = 0,
    BLACK = 1,
};
int32_t active_features[2][MAX_ACTIVE_FEATURE];
const uint8_t nn_indices[2][12] = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5}};

int bitScanForward(uint64_t bb)
{
    return __builtin_ctzll(bb);
}
int poplsb(uint64_t *bb)
{
    int lsb = bitScanForward(*bb);
    *bb &= *bb - 1;
    return lsb;
}
int read_position(Data *data)
{
    int num = 0, sq = 0, piece = 0;

    uint8_t wking = data->wking;
    uint8_t bking = data->bking;

    while (data->occupied)
    {
        sq = poplsb(&data->occupied);
        if (num % 2)
        {
            piece = data->packed[num / 2] >> 4;
            active_features[BLACK][num]   = nn_indices[BLACK][piece] * 64 + Mirror(sq);
            active_features[WHITE][num++] = piece * 64 + sq;
        }
        else
        {
            piece = data->packed[num / 2] & 15;
            active_features[BLACK][num] = nn_indices[BLACK][piece] * 64 + Mirror(sq);
            active_features[WHITE][num++] = piece * 64 + sq;
        }
    }

    active_features[BLACK][num] = 5 * 64 + Mirror(bking);
    active_features[WHITE][num++] = 11 * 64 + bking;

    active_features[BLACK][num] = 11 * 64 + Mirror(wking);
    active_features[WHITE][num++] = 5 * 64 + wking;

    return num;
}
void load_data(Data *data, int number_of_data, int32_t *feature_indices_us, int32_t *feature_indices_enemy, float *target)
{
    int k = 0;
    for (int i = 0; i < number_of_data; i++)
    {
        int num = read_position(&data[i]);
        for (int j = 0; j < num; j++)
        {
            feature_indices_us[MAX_ACTIVE_FEATURE * i + j] = active_features[data[i].side][j];
            feature_indices_enemy[MAX_ACTIVE_FEATURE * i + j] = active_features[!data[i].side][j];
        }
        for(int j=num ; j<MAX_ACTIVE_FEATURE ; j++)
        {
            feature_indices_us[MAX_ACTIVE_FEATURE * i + j] = -1;
            feature_indices_enemy[MAX_ACTIVE_FEATURE * i + j] = -1;
        }
        target[i] = data[i].eval;
        k++;
    }
}
#endif