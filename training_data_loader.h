#ifndef _DATA_LOADER_H_
#define _DATA_LOADER_H_
#include "activations/activations.h"

//BulletFormat
struct Board
{
    uint64_t occupancy; // 8 byte
    uint8_t pieces[16]; // 16 bytes
    int16_t score; // 2 bytes
    uint8_t result; // 1 byte
    uint8_t king_square; // 1 byte
    uint8_t opp_king_square; // 1 byte
    uint8_t padding[3]; // 3 bytes padding
};

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
const uint8_t nn_indices[2][14] = {{0, 1, 2, 3, 4, 5, 0,0,6, 7, 8, 9, 10, 11}, {6, 7, 8, 9, 10, 11,0,0, 0, 1, 2, 3, 4, 5}};

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
int nn_index( int piece, int sq, int side)
{
    if (side == 1)
    {
        sq = Mirror(sq);
    }
    return nn_indices[side][piece] * 64 + sq;
}
int read_position(Board *data)
{
    int num = 0, sq = 0, piece = 0;

    auto king_square = data->king_square;
    auto opp_king_square  = data->opp_king_square;

    while (data->occupancy)
    {
        auto temp = data->occupancy;
        sq = poplsb(&data->occupancy);
        if (num % 2)
            piece = data->pieces[num / 2] >> 4;
        else
            piece = data->pieces[num / 2] & 15;

        active_features[0][num]  = nn_index(piece, sq, 0);
        active_features[1][num++] = nn_index(piece, sq, 1);
    }

    return num;
}
void load_data(Board *data, int number_of_data, int32_t *feature_indices_us, int32_t *feature_indices_enemy, float *cp, float *wdl)
{
    for (int i = 0; i < number_of_data; i++)
    {
        int num = read_position(&data[i]);
        for (int j = 0; j < num; j++)
        {
            feature_indices_us[MAX_ACTIVE_FEATURE * i + j] = active_features[0][j];
            feature_indices_enemy[MAX_ACTIVE_FEATURE * i + j] = active_features[1][j];
        }
        for (int j = num; j < MAX_ACTIVE_FEATURE; j++)
        {
            feature_indices_us[MAX_ACTIVE_FEATURE * i + j] = -1;
            feature_indices_enemy[MAX_ACTIVE_FEATURE * i + j] = -1;
        }
        cp[i] = data[i].score;
        wdl[i] = (data[i].result) / 2.0f;
    }
}
#endif