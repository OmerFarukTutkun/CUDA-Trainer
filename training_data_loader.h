#ifndef _DATA_LOADER_H_
#define _DATA_LOADER_H_
#include "activations/activations.h"

typedef struct Data
{
    uint64_t occupied; // occupancy without kings, A1 = 0
    int16_t eval;
    int8_t result; // -1 , 0 , 1
    uint8_t side;  // 1 for black 0 for white
    uint8_t wking; // king square
    uint8_t bking;
    uint8_t packed[15]; // 2 pieces in one byte
} Data;

#define Mirror(sq) (sq ^ 56)
#define HorizontalMirror(sq) (sq ^ 7)

enum Pieces
{
    PAWN = 0,
    KNIGHT = 1,
    BISHOP = 2,
    ROOK = 3,
    QUEEN = 4,
    KING = 5,
    BLACK_PAWN = 6,
    BLACK_KNIGHT = 7,
    BLACK_BISHOP = 8,
    BLACK_ROOK = 9,
    BLACK_QUEEN = 10,
    BLACK_KING = 11,
};

enum Colors
{
    WHITE = 0,
    BLACK = 1,
};
int king_indices[64]{
    0, 1, 2, 3, 3, 2, 1, 0,
    4, 5, 6, 7, 7, 6, 5, 4,
    8, 9, 10, 11, 11, 10, 9, 8,
    12, 13, 14, 15, 15, 14, 13, 12,
    16, 17, 18, 19, 19, 18, 17, 16,
    20, 21, 22, 23, 23, 22, 21, 20,
    24, 25, 26, 27, 27, 26, 25, 24,
    28, 29, 30, 31, 31, 30, 29, 28};

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
int nn_index(int king, int piece, int sq, int side)
{
    if (side == BLACK)
    {
        sq = Mirror(sq);
        king = Mirror(king);
    }
    if (king % 8 < 4)
        return king_indices[king] * 768 + nn_indices[side][piece] * 64 + HorizontalMirror(sq);
    else
        return king_indices[king] * 768 + nn_indices[side][piece] * 64 + sq;
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
            piece = data->packed[num / 2] >> 4;
        else
            piece = data->packed[num / 2] & 15;

        active_features[BLACK][num] = nn_index(bking, piece, sq, BLACK);
        active_features[WHITE][num++] = nn_index(wking, piece, sq, WHITE);
    }

    active_features[BLACK][num] = nn_index(bking, BLACK_KING, bking, BLACK);
    active_features[WHITE][num++] = nn_index(wking, BLACK_KING, bking, WHITE);

    active_features[BLACK][num] = nn_index(bking, KING, wking, BLACK);
    active_features[WHITE][num++] = nn_index(wking, KING, wking, WHITE);

    return num;
}
void load_data(Data *data, int number_of_data, int32_t *feature_indices_us, int32_t *feature_indices_enemy, float *cp, float *wdl)
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
        for (int j = num; j < MAX_ACTIVE_FEATURE; j++)
        {
            feature_indices_us[MAX_ACTIVE_FEATURE * i + j] = -1;
            feature_indices_enemy[MAX_ACTIVE_FEATURE * i + j] = -1;
        }
        cp[i] = data[i].eval;
        wdl[i] = (data[i].result + 1) / 2.0;
        k++;
    }
}
#endif