// hyperion/src/cpp/core/bitboard.cpp
#include "bitboard.hpp"
#include "constants.hpp"
#include "position.hpp"
#include <immintrin.h> // Required for _pext_u64 <-- comments about this use below
#include <algorithm>
#include <vector>

/*The _pext_u64 (Parallel Bit Extract) instruction is part of the BMI2 instruction set available on modern x86 CPUs. 
It's highly effective for magic bitboard implementations because it can directly map the bits of occupied squares
on a "mask" to a dense index, replacing the traditional magic multiplication and shift.*/

#if (defined(__GNUC__) || defined(__clang__)) && defined(__BMI2__)
    #define USE_BMI2_SLIDERS 1
    //#error "BMI2 for GCC/Clang path taken"
#elif defined(_MSC_VER) && defined(__AVX2__) // /arch:AVX2 generally means BMI2 is available
    #define USE_BMI2_SLIDERS 1
    //#error "BMI2 for MSVC path taken (_AVX2_ defined)"
#else
    #define USE_BMI2_SLIDERS
#endif

// namespace is used for organization
namespace hyperion { 
namespace core {

// Sources
// https://en.wikipedia.org/wiki/Intrinsic_function <- look for more info on general Intrinsic functions
// https://www.chessprogramming.org/BitScan <- bitscan info

/*
Compilers for C and C++, of Microsoft, Intel, and the GNU Compiler Collection (GCC) implement intrinsics
that map directly to the x86 single instruction. Intrinsics allow mapping to standard assembly instructions
that are not normally accessible through C/C++, e.g., bit scan.

Dev Note: Im going to try to use them for there performace gain, but if things get too
messy or weird. I will just remove them and then update functions. For now I will elif because I dont
know if we use clang or GCC (ive used GCC more in the past). In the end, higher performace = higher elo
*/

#if defined(_MSC_VER)
    #include <intrin.h> // For _BitScanForward64, __popcnt64
#elif defined(__GNUC__) || defined(__clang__)
    // GCC and Clang have built-in functions (no specific include needed for __builtin_popcountll, __builtin_ctzll)
#else
    // Fallback implementations if no intrinsics, we should TRY use them though for performance gain
#endif

// --- Core Bitboard Operations ---

//--
/* set_bit (int version) */
//--
// Sets the bit at the given square_index in the bitboard.
// The square_index is 0-63, corresponding to A1-H8.
// Modifies the bitboard in place.
// Performs a bounds check on square_index.
void set_bit(bitboard_t& bb, int square_index) {
    if(square_index >= 0 && square_index < hyperion::core::NUM_SQUARES) { //bounds check
        bb |= (1ULL << square_index); // unsigned long long
    }
}

//--
/* clear_bit (int version) */
//--
// Clears the bit at the given square_index in the bitboard.
// The square_index is 0-63, corresponding to A1-H8.
// Modifies the bitboard in place.
// Performs a bounds check on square_index.
void clear_bit(bitboard_t& bb, int square_index) {
    if(square_index >= 0 && square_index < hyperion::core::NUM_SQUARES) { //bounds check
        bb &= ~(1ULL << square_index); // unsigned long long
    }
}

//--
/* get_bit (int version) */
//--
// Checks if the bit at the given square_index in the bitboard is set.
// The square_index is 0-63, corresponding to A1-H8.
// Returns true if the bit is set, false otherwise.
// Performs a bounds check on square_index.
bool get_bit(bitboard_t bb, int square_index) {
    if (square_index < 0 || square_index >= hyperion::core::NUM_SQUARES) { //bounds check
        return false;
    }
    return (bb & (1ULL << square_index)) != 0;
}

// --- Utility Functions ---

//--
/* print_bitboard */
//--
// Prints a visual representation of the bitboard to the console.
// '1' represents a set bit, '.' (EMPTY_SQUARE_CHAR) represents a clear bit.
// Ranks are printed 8 down to 1, files a to h.
// Also prints the hexadecimal value of the bitboard.
void print_bitboard(bitboard_t bb) {
    std::cout << " Bitboard 0x" << std::hex << bb << std::dec << std::endl;
    for (int rank_idx = 7; rank_idx >= 0; --rank_idx) { //ranks 8 down to 1
        std::cout << (rank_idx + 1) << "  ";
        for (int file_idx = 0; file_idx < 8; ++file_idx){ // Files A to H
            // This is where the square index calculation using the global
            // constants in constant.hpp are helpful.
            // How to get the index num? general formal is right here:
            //     int square_index = rank_idx * 8 + file_idx;
            // Then go through and get_bit(bb, square_index) through all 64 indexs
            int current_square_idx = rank_idx * 8 + file_idx;
            if (get_bit(bb,current_square_idx)) {
                std::cout << "1 ";
            }
            else {
                std::cout << hyperion::core::EMPTY_SQUARE_CHAR << " "; //
            }
        } 
        std::cout << std::endl;
    } 
    std::cout << "   a b c d e f g h" << std::endl; 
    std::cout << std::endl;

    // example empty bitboard:
    //
    // 8  . . . . . . . .
    // 7  . . . . . . . .
    // 6  . . . . . . . .
    // 5  . . . . . . . .
    // 4  . . . . . . . .
    // 3  . . . . . . . .
    // 2  . . . . . . . .
    // 1  . . . . . . . .
    //    a b c d e f g h
}

//--
/* count_set_bits */
//--
// Counts the number of set bits (population count) in a given bitboard.
// Uses compiler intrinsics (__popcnt64 for MSVC, __builtin_popcountll for GCC/Clang) for performance if available.
// Falls back to Brian Kernighan's Algorithm if intrinsics are not available.
// Returns the total number of set bits.
int count_set_bits(bitboard_t bb) {
    // this is where possible optimization is potentially used
    // first try _MSC_VER, second try GCC or CLANG, third fallback is
    // Brain Kernighans Algorithm:
    //     https://www.techiedelight.com/brian-kernighans-algorithm-count-set-bits-integer/
    //
    //     for more info on that algorithm read above^
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt64(bb));
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(bb);
#else
    // Brian Kernighan's Algorithm (fallback/ failsafe)
    int count = 0;
    while (bb > 0) {
        bb &= (bb - 1);
        count++;
    }
    return count;
#endif
}

//--
/* get_lsb_index */
//--
// Finds the index of the least significant bit (LSB) that is set in the bitboard.
// Uses compiler intrinsics (_BitScanForward64 for MSVC, __builtin_ctzll for GCC/Clang) for performance if available.
// Falls back to a manual search if intrinsics are not available.
// Returns the 0-63 index of the LSB.
// Returns static_cast<int>(hyperion::core::square_e::NO_SQ) if the bitboard is empty.
int get_lsb_index(bitboard_t bb) {
    if (bb == 0){
        return static_cast<int>(hyperion::core::square_e::NO_SQ);
    }
#if defined(_MSC_VER)
    unsigned long index;
    _BitScanForward64(&index, bb);
    return static_cast<int>(index);
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(bb); // Count Trailing Zeros
#else
    // Fallback/ failsafe:
    int count = 0;
    // This loop assumes bb != 0 (guaranteed by the check above)
    while (!((bb >> count) & 1ULL)) {
        count++;
    }
    return count;
#endif
}

//--
/* pop_lsb */
//--
// Finds the index of the least significant bit (LSB), clears it from the bitboard, and returns the index.
// Modifies the input bitboard `bb` by clearing the LSB.
// Internally uses get_lsb_index to find the LSB.
// Uses an efficient method (bb &= (bb - 1)) to clear the LSB.
// Returns the 0-63 index of the LSB that was popped.
// Returns static_cast<int>(hyperion::core::square_e::NO_SQ) if the bitboard was initially empty.
int pop_lsb(bitboard_t& bb) {
    int lsb_idx = get_lsb_index(bb); 
    if (lsb_idx != static_cast<int>(hyperion::core::square_e::NO_SQ)) {
        // this should be faster then calling the clear_bit()
        // LSB: bb &= (bb -1); <-- should be faster if LSB was
        // found via intrinsics
        // If we decide not to use intrinsics, this can all be chaned to just call
        // clear_bit
        bb &= (bb - 1); // Efficiently clear LSB
    }
    return lsb_idx;
}

//--
/* square_to_algebraic */
//--
// Converts a 0-63 square index to its algebraic notation (e.g., 0 to "a1", 63 to "h8").
// Performs a bounds check on square_index.
// Returns "??" for an invalid square index.
// Returns a std::string representing the algebraic notation.
std::string square_to_algebraic(int square_index) {
    if (square_index < 0 || square_index >= hyperion::core::NUM_SQUARES) {
        return "??"; //bounds check, this would be a completly invalid square
    }
    char file_char = 'a' + (square_index % 8); 
    char rank_char = '1' + (square_index / 8); 
    std::string s = "";
    s += file_char;
    s += rank_char;
    return s;
}

// --- attacks ---

// Define the actual storage for the extern declared arrays
std::array<std::array<bitboard_t, NUM_SQUARES>, 2> pawn_attacks;
std::array<bitboard_t, NUM_SQUARES> knight_attacks;
std::array<bitboard_t, NUM_SQUARES> king_attacks;

// Magic Bitboard table definitions
MagicEntry rook_magic_entries[NUM_SQUARES];
MagicEntry bishop_magic_entries[NUM_SQUARES];

bitboard_t rook_attack_table[ROOK_ATTACK_TABLE_SIZE];
bitboard_t bishop_attack_table[BISHOP_ATTACK_TABLE_SIZE];

//--
/* add_attack (internal helper) */
//--
// Internal helper function used during attack table initialization.
// Safely adds an attack to the `attacks_bb` if the target square, calculated from
// `current_file`, `current_rank`, `df` (delta file), and `dr` (delta rank), is on the board.
// Modifies `attacks_bb` in place.
void add_attack(bitboard_t& attacks_bb, int current_file, int current_rank, int df, int dr) {
    int to_file = current_file + df;
    int to_rank = current_rank + dr;
    if (to_file >= 0 && to_file < 8 && to_rank >= 0 && to_rank < 8) {
        set_bit(attacks_bb, static_cast<square_e>(to_rank * 8 + to_file));
    }
}

// --- Helper: Slow but correct attack generation for initializing magic tables ---
// 
// These are only used during initialize_attack_tables()

//--
/* generate_attacks_slow_internal (internal helper) */
//--
// Generates slider attacks (rook or bishop) for a given square, considering blockers.
// This is a slow, reference implementation used for populating magic bitboard attack tables.
// `sq`: The 0-63 index of the square from which to generate attacks.
// `blockers`: A bitboard representing all blocking pieces on the board.
// `is_rook`: True if generating rook attacks, false for bishop attacks.
// Returns a bitboard of all attacked squares.
bitboard_t generate_attacks_slow_internal(int sq, bitboard_t blockers, bool is_rook) {
    bitboard_t attacks = EMPTY_BB; // Assuming EMPTY_BB is 0ULL
    int r = sq / 8;
    int f = sq % 8;

    const int dr_rook[] = {-1, 1, 0, 0}; 
    const int df_rook[] = {0, 0, -1, 1};
    const int dr_bishop[] = {-1, -1, 1, 1}; 
    const int df_bishop[] = {-1, 1, -1, 1};
    int num_dirs = 4;

    for (int i = 0; i < num_dirs; ++i) {
        int current_dr = is_rook ? dr_rook[i] : dr_bishop[i];
        int current_df = is_rook ? df_rook[i] : df_bishop[i];
        for (int j = 1; j < 8; ++j) { 
            int nr = r + current_dr * j;
            int nf = f + current_df * j;
            if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) {
                set_bit(attacks, nr * 8 + nf); // Ensure set_bit is accessible
                if (get_bit(blockers, nr * 8 + nf)) { // Ensure get_bit is accessible
                    break;
                }
            } else {
                break; 
            }
        }
    }
    return attacks;
}

//--
/* get_blocker_permutations (internal helper) */
//--
// Generates all possible blocker configurations (permutations) for a given mask.
// `mask`: A bitboard representing the relevant squares for blockers (e.g., rook or bishop mask for magic bitboards).
// Returns a std::vector of bitboards, where each bitboard is one permutation of blockers within the mask.
// This is used during the initialization of magic bitboard attack tables.

std::vector<bitboard_t> get_blocker_permutations(bitboard_t mask) {
    std::vector<bitboard_t> permutations;
#if defined(USE_BMI2_SLIDERS) // Or a separate macro for PDEP if preferred
    int num_set_bits = count_set_bits(mask); // Uses your existing popcount function
    for (uint64_t i = 0; i < (1ULL << num_set_bits); ++i) {
        permutations.push_back(_pdep_u64(i, mask));
    }
#else
    // Original implementation (fallback)
    std::vector<int> set_bits_indices;
    for (int i = 0; i < NUM_SQUARES; ++i) {
        if (get_bit(mask, i)) {
            set_bits_indices.push_back(i);
        }
    }
    int num_set_bits = set_bits_indices.size();
    for (int i = 0; i < (1 << num_set_bits); ++i) {
        bitboard_t current_permutation = EMPTY_BB;
        for (int j = 0; j < num_set_bits; ++j) {
            if ((i >> j) & 1) {
                set_bit(current_permutation, set_bits_indices[j]);
            }
        }
        permutations.push_back(current_permutation);
    }
#endif
    return permutations;
}

//---------------------------//
//                           //
// PRECOMPUTED ATTACKS BELOW //
//                           //
//---------------------------//

namespace {

#ifndef USE_BMI2_SLIDERS
const uint64_t PRECOMPUTED_ROOK_MAGICS[64] = {
  0x80102040008000ULL,   0x4040001000200040ULL,   0x8880200209801000ULL,   0x2000600a0384050ULL,
  0x3c80040082480080ULL,   0x9100080100040082ULL,   0x1000d8100046200ULL,   0x21000100142081c2ULL,
  0xa0800030884000ULL,   0x10802000804000ULL,   0x282001028804200ULL,   0x181001000082100ULL,
  0x4000800400800800ULL,   0x2200800400020080ULL,   0x2402000802008104ULL,   0xa00100010000906aULL,
  0x8080010021004088ULL,   0x20008020804000ULL,   0x1020008028801000ULL,   0x622020040081020ULL,
  0x80808008000400ULL,   0x4800808002000400ULL,   0x4042040008414210ULL,   0xc020000608f04ULL,
  0x9064400080088020ULL,   0x20400080802000ULL,   0x20002100410012ULL,   0x1000900201004ULL,
  0x20080080800400ULL,   0x20080040080ULL,   0x31000100040200ULL,   0x8000144a00008524ULL,
  0x2004102002080ULL,   0x4000402008401001ULL,   0x8002200043001100ULL,   0x100009002100ULL,
  0x8110080080800400ULL,   0x88800400800201ULL,   0x2800010204001008ULL,   0x40000a042000411ULL,
  0xa400400080028020ULL,   0x900810040010030ULL,   0x8482001010012ULL,   0x44220040120009ULL,
  0x68008509010010ULL,   0x184020004008080ULL,   0x8830089002040001ULL,   0x810007100820004ULL,
  0x22a0801021004900ULL,   0xf28520100802200ULL,   0x10102000410100ULL,   0x5005002010000900ULL,
  0x8440040008008080ULL,   0x1200800200040080ULL,   0x401000402000100ULL,   0x80304401048200ULL,
  0x100800100204011ULL,   0x1881002880401202ULL,   0x8306a00a00308042ULL,   0x6000600815013001ULL,
  0x11220010042048aaULL,   0x2022000408015082ULL,   0x10501a82082104ULL,   0x1098108900240042ULL,
};
#endif

const uint64_t PRECOMPUTED_ROOK_MASKS[64] = {
  0x101010101017eULL,   0x202020202027cULL,   0x404040404047aULL,   0x8080808080876ULL,
  0x1010101010106eULL,   0x2020202020205eULL,   0x4040404040403eULL,   0x8080808080807eULL,
  0x1010101017e00ULL,   0x2020202027c00ULL,   0x4040404047a00ULL,   0x8080808087600ULL,
  0x10101010106e00ULL,   0x20202020205e00ULL,   0x40404040403e00ULL,   0x80808080807e00ULL,
  0x10101017e0100ULL,   0x20202027c0200ULL,   0x40404047a0400ULL,   0x8080808760800ULL,
  0x101010106e1000ULL,   0x202020205e2000ULL,   0x404040403e4000ULL,   0x808080807e8000ULL,
  0x101017e010100ULL,   0x202027c020200ULL,   0x404047a040400ULL,   0x8080876080800ULL,
  0x1010106e101000ULL,   0x2020205e202000ULL,   0x4040403e404000ULL,   0x8080807e808000ULL,
  0x1017e01010100ULL,   0x2027c02020200ULL,   0x4047a04040400ULL,   0x8087608080800ULL,
  0x10106e10101000ULL,   0x20205e20202000ULL,   0x40403e40404000ULL,   0x80807e80808000ULL,
  0x17e0101010100ULL,   0x27c0202020200ULL,   0x47a0404040400ULL,   0x8760808080800ULL,
  0x106e1010101000ULL,   0x205e2020202000ULL,   0x403e4040404000ULL,   0x807e8080808000ULL,
  0x7e010101010100ULL,   0x7c020202020200ULL,   0x7a040404040400ULL,   0x76080808080800ULL,
  0x6e101010101000ULL,   0x5e202020202000ULL,   0x3e404040404000ULL,   0x7e808080808000ULL,
  0x7e01010101010100ULL,   0x7c02020202020200ULL,   0x7a04040404040400ULL,   0x7608080808080800ULL,
  0x6e10101010101000ULL,   0x5e20202020202000ULL,   0x3e40404040404000ULL,   0x7e80808080808000ULL,
};

const int PRECOMPUTED_ROOK_BITS[64] = {
  12,   11,   11,   11,   11,   11,   11,   12,   11,   10,   10,   10,   10,   10,   10,   11,
  11,   10,   10,   10,   10,   10,   10,   11,   11,   10,   10,   10,   10,   10,   10,   11,
  11,   10,   10,   10,   10,   10,   10,   11,   11,   10,   10,   10,   10,   10,   10,   11,
  11,   10,   10,   10,   10,   10,   10,   11,   12,   11,   11,   11,   11,   11,   11,   12,
};

#ifndef USE_BMI2_SLIDERS
const uint8_t PRECOMPUTED_ROOK_SHIFTS[64] = {
  52,   53,   53,   53,   53,   53,   53,   52,   53,   54,   54,   54,   54,   54,   54,   53,
  53,   54,   54,   54,   54,   54,   54,   53,   53,   54,   54,   54,   54,   54,   54,   53,
  53,   54,   54,   54,   54,   54,   54,   53,   53,   54,   54,   54,   54,   54,   54,   53,
  53,   54,   54,   54,   54,   54,   54,   53,   52,   53,   53,   53,   53,   53,   53,   52,
};
#endif
const uint32_t PRECOMPUTED_ROOK_OFFSETS[64] = {
       0,     4096,     6144,     8192,    10240,    12288,    14336,    16384,
   20480,    22528,    23552,    24576,    25600,    26624,    27648,    28672,
   30720,    32768,    33792,    34816,    35840,    36864,    37888,    38912,
   40960,    43008,    44032,    45056,    46080,    47104,    48128,    49152,
   51200,    53248,    54272,    55296,    56320,    57344,    58368,    59392,
   61440,    63488,    64512,    65536,    66560,    67584,    68608,    69632,
   71680,    73728,    74752,    75776,    76800,    77824,    78848,    79872,
   81920,    86016,    88064,    90112,    92160,    94208,    96256,    98304,
};

// --- BISHOP DATA ---
#ifndef USE_BMI2_SLIDERS
const uint64_t PRECOMPUTED_BISHOP_MAGICS[64] = {
  0x9c0010104008082ULL,   0x2004414821050000ULL,   0x14040192004001ULL,   0x8044404080100002ULL,
  0x24102880090002ULL,   0x1202080484c00000ULL,   0x4029081124202080ULL,   0xb12020206196410ULL,
  0x40c00808888090ULL,   0x2010228801040090ULL,   0x100411400808000ULL,   0x8011044044820000ULL,
  0x4801c0420002021ULL,   0x6020802080902ULL,   0x809908c05201004ULL,   0x80010110908400ULL,
  0x40000484080a04ULL,   0x4ea0040c042404ULL,   0x40a086042840080ULL,   0x644202812002004ULL,
  0x8002002c00a22004ULL,   0x810400201103100ULL,   0x8024212290880810ULL,   0x1a008500820104ULL,
  0xa200801e0188320ULL,   0x101010041400a5ULL,   0x80400c8002029ULL,   0x2040800c4202040ULL,
  0x1010000104010ULL,   0xa018060482002ULL,   0x8048003048831ULL,   0x6901002806028402ULL,
  0x80c4100804400280ULL,   0x2100290300208ULL,   0x2011000810044ULL,   0x2401404800028200ULL,
  0x40008020120020ULL,   0x1104240220841000ULL,   0x204144042448800ULL,   0x108020820124114ULL,
  0x10422130c0000880ULL,   0x302c02080b004480ULL,   0x42d884058001008ULL,   0x1000c204812801ULL,
  0x218080104000840ULL,   0x1440710049000080ULL,   0x8300084800200ULL,   0x9448424042001440ULL,
  0x400429080a100029ULL,   0x601004a10040812ULL,   0x8800044404110000ULL,   0x1000004042020000ULL,
  0x8003102042048020ULL,   0x80a00810016280ULL,   0x450101081184204ULL,   0x20040080b1000aULL,
  0xc2020250c104048ULL,   0x400008041682000ULL,   0x1000422605108800ULL,   0x300100011420210ULL,
  0x2004000090120211ULL,   0xc000808084825ULL,   0x204050204a042042ULL,   0x120200c05005013ULL,
};
#endif
const uint64_t PRECOMPUTED_BISHOP_MASKS[64] = {
  0x40201008040200ULL,   0x402010080400ULL,   0x4020100a00ULL,   0x40221400ULL,
  0x2442800ULL,   0x204085000ULL,   0x20408102000ULL,   0x2040810204000ULL,
  0x20100804020000ULL,   0x40201008040000ULL,   0x4020100a0000ULL,   0x4022140000ULL,
  0x244280000ULL,   0x20408500000ULL,   0x2040810200000ULL,   0x4081020400000ULL,
  0x10080402000200ULL,   0x20100804000400ULL,   0x4020100a000a00ULL,   0x402214001400ULL,
  0x24428002800ULL,   0x2040850005000ULL,   0x4081020002000ULL,   0x8102040004000ULL,
  0x8040200020400ULL,   0x10080400040800ULL,   0x20100a000a1000ULL,   0x40221400142200ULL,
  0x2442800284400ULL,   0x4085000500800ULL,   0x8102000201000ULL,   0x10204000402000ULL,
  0x4020002040800ULL,   0x8040004081000ULL,   0x100a000a102000ULL,   0x22140014224000ULL,
  0x44280028440200ULL,   0x8500050080400ULL,   0x10200020100800ULL,   0x20400040201000ULL,
  0x2000204081000ULL,   0x4000408102000ULL,   0xa000a10204000ULL,   0x14001422400000ULL,
  0x28002844020000ULL,   0x50005008040200ULL,   0x20002010080400ULL,   0x40004020100800ULL,
  0x20408102000ULL,   0x40810204000ULL,   0xa1020400000ULL,   0x142240000000ULL,
  0x284402000000ULL,   0x500804020000ULL,   0x201008040200ULL,   0x402010080400ULL,
  0x2040810204000ULL,   0x4081020400000ULL,   0xa102040000000ULL,   0x14224000000000ULL,
  0x28440200000000ULL,   0x50080402000000ULL,   0x20100804020000ULL,   0x40201008040200ULL,
};

const int PRECOMPUTED_BISHOP_BITS[64] = {
   6,    5,    5,    5,    5,    5,    5,    6,    5,    5,    5,    5,    5,    5,    5,    5,
   5,    5,    7,    7,    7,    7,    5,    5,    5,    5,    7,    9,    9,    7,    5,    5,
   5,    5,    7,    9,    9,    7,    5,    5,    5,    5,    7,    7,    7,    7,    5,    5,
   5,    5,    5,    5,    5,    5,    5,    5,    6,    5,    5,    5,    5,    5,    5,    6,
};
#ifndef USE_BMI2_SLIDERS
const uint8_t PRECOMPUTED_BISHOP_SHIFTS[64] = {
  58,   59,   59,   59,   59,   59,   59,   58,   59,   59,   59,   59,   59,   59,   59,   59,
  59,   59,   57,   57,   57,   57,   59,   59,   59,   59,   57,   55,   55,   57,   59,   59,
  59,   59,   57,   55,   55,   57,   59,   59,   59,   59,   57,   57,   57,   57,   59,   59,
  59,   59,   59,   59,   59,   59,   59,   59,   58,   59,   59,   59,   59,   59,   59,   58,
};
#endif
const uint32_t PRECOMPUTED_BISHOP_OFFSETS[64] = {
      0,      64,      96,     128,     160,     192,     224,     256,
    320,     352,     384,     416,     448,     480,     512,     544,
    576,     608,     640,     768,     896,    1024,    1152,    1184,
   1216,    1248,    1280,    1408,    1920,    2432,    2560,    2592,
   2624,    2656,    2688,    2816,    3328,    3840,    3968,    4000,
   4032,    4064,    4096,    4224,    4352,    4480,    4608,    4640,
   4672,    4704,    4736,    4768,    4800,    4832,    4864,    4896,
   4928,    4992,    5024,    5056,    5088,    5120,    5152,    5184,

};
};

//---------------------------//
//                           //
// PRECOMPUTED ATTACKS ABOVE //
//                           //
//---------------------------//

//--
/* initialize_attack_tables */
//--
// Initializes all precomputed attack tables: pawn, knight, king, and slider (rook/bishop) attacks.
// Pawn, knight, and king attacks are generated directly.
// Rook and bishop attacks are generated using magic bitboards:
//   - Sets up MagicEntry structures for each square using precomputed magics, masks, shifts, and offsets.
//   - For each square and each blocker permutation within its mask, it calculates the magic index
//     and stores the slowly generated attack set (using `generate_attacks_slow_internal`)
//     into the global `rook_attack_table` or `bishop_attack_table`.
// This function must be called once at program startup before any attack lookups are performed.
void initialize_attack_tables() {
    
    // 1. Initialize Pawn, Knight, King attacks
    for (int sq_idx = 0; sq_idx < NUM_SQUARES; ++sq_idx) { //
        int rank = sq_idx / 8;
        int file = sq_idx % 8;

        // Pawn attacks
        pawn_attacks[WHITE][sq_idx] = EMPTY_BB; //
        if (rank < 7) { // White pawns can't attack from 8th rank
            if (file > 0) set_bit(pawn_attacks[WHITE][sq_idx], static_cast<square_e>((rank + 1) * 8 + (file - 1))); // Capture left
            if (file < 7) set_bit(pawn_attacks[WHITE][sq_idx], static_cast<square_e>((rank + 1) * 8 + (file + 1))); // Capture right
        }
        pawn_attacks[BLACK][sq_idx] = EMPTY_BB; //
        if (rank > 0) { // Black pawns can't attack from 1st rank
            if (file > 0) set_bit(pawn_attacks[BLACK][sq_idx], static_cast<square_e>((rank - 1) * 8 + (file - 1))); // Capture left
            if (file < 7) set_bit(pawn_attacks[BLACK][sq_idx], static_cast<square_e>((rank - 1) * 8 + (file + 1))); // Capture right
        }

        // Knight attacks
        knight_attacks[sq_idx] = EMPTY_BB;
        const int knight_moves[8][2] = {
            {1, 2}, {1, -2}, {-1, 2}, {-1, -2},
            {2, 1}, {2, -1}, {-2, 1}, {-2, -1}
        };
        for (const auto& move : knight_moves) {
            add_attack(knight_attacks[sq_idx], file, rank, move[0], move[1]);
        }

        // King attacks
        king_attacks[sq_idx] = EMPTY_BB;
        const int king_moves[8][2] = {
            {0, 1}, {0, -1}, {1, 0}, {-1, 0},
            {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
        };
        for (const auto& move : king_moves) {
            add_attack(king_attacks[sq_idx], file, rank, move[0], move[1]);
        }
    }
    // Initialize Rook Magic Entries and Rook Attack Table
    for (int sq_idx = 0; sq_idx < NUM_SQUARES; ++sq_idx) {
        rook_magic_entries[sq_idx].mask = PRECOMPUTED_ROOK_MASKS[sq_idx];
        rook_magic_entries[sq_idx].attacks = rook_attack_table + PRECOMPUTED_ROOK_OFFSETS[sq_idx];
    #ifndef USE_BMI2_SLIDERS
        rook_magic_entries[sq_idx].magic_number = PRECOMPUTED_ROOK_MAGICS[sq_idx];
        rook_magic_entries[sq_idx].shift = PRECOMPUTED_ROOK_SHIFTS[sq_idx];
    #endif

        std::vector<bitboard_t> permutations = get_blocker_permutations(rook_magic_entries[sq_idx].mask);
        for (const auto& blockers : permutations) {
    #ifdef USE_BMI2_SLIDERS
            // 'blockers' is already a submask of entry.mask, so _pext_u64(blockers, entry.mask) == blockers_compacted
            // The value 'blockers' from get_blocker_permutations (if using PDEP) is already a direct submask.
            // _pext_u64(blockers, rook_magic_entries[sq_idx].mask) extracts the bits of 'blockers'
            // that are within the mask and compacts them. This is the desired index.
            uint64_t pext_index = _pext_u64(blockers, rook_magic_entries[sq_idx].mask);
            rook_magic_entries[sq_idx].attacks[pext_index] = generate_attacks_slow_internal(sq_idx, blockers, true);
    #else
            uint64_t magic_index = (blockers * rook_magic_entries[sq_idx].magic_number) >> rook_magic_entries[sq_idx].shift;
            rook_magic_entries[sq_idx].attacks[magic_index] = generate_attacks_slow_internal(sq_idx, blockers, true);
    #endif
        }
        // Debug code that checks if the magic number = mask
        // couts bitboard for each magic number as  well
        /*
        bool square_ok = true;
        for (const auto& blockers_on_mask : permutations) {
            uint64_t magic_index = (blockers_on_mask * rook_magic_entries[sq_idx].magic_number) >> rook_magic_entries[sq_idx].shift;
            bitboard_t attacks_from_slow_gen = generate_attacks_slow_internal(sq_idx, blockers_on_mask, true);
            rook_magic_entries[sq_idx].attacks[magic_index] = attacks_from_slow_gen;
        }
        for (const auto& blockers_on_mask : permutations) {
            uint64_t magic_index = (blockers_on_mask * rook_magic_entries[sq_idx].magic_number) >> rook_magic_entries[sq_idx].shift;
            bitboard_t attacks_from_table = rook_magic_entries[sq_idx].attacks[magic_index];
            bitboard_t attacks_from_slow_gen = generate_attacks_slow_internal(sq_idx, blockers_on_mask, true);
            if (attacks_from_table != attacks_from_slow_gen) {
                std::cout << "VERIFY FAIL (Rook Sq " << sq_idx << ", mask " << std::hex << rook_magic_entries[sq_idx].mask << std::dec
                      << "): For blockers " << std::hex << blockers_on_mask << std::dec << std::endl;
                std::cout << "  Table  : " << std::hex << attacks_from_table << std::dec << std::endl;
                print_bitboard(attacks_from_table);
                std::cout << "  SlowGen: " << std::hex << attacks_from_slow_gen << std::dec << std::endl;
                print_bitboard(attacks_from_slow_gen);
                square_ok = false;
            }
        }
        if (!square_ok) {
        std::cout << "Rook magic for square " << sq_idx << " has issues." << std::endl;
        }
        */
    }

    // Initialize Bishop Magic Entries and Bishop Attack Table
    for (int sq_idx = 0; sq_idx < NUM_SQUARES; ++sq_idx) {
        bishop_magic_entries[sq_idx].mask = PRECOMPUTED_BISHOP_MASKS[sq_idx];
        bishop_magic_entries[sq_idx].attacks = bishop_attack_table + PRECOMPUTED_BISHOP_OFFSETS[sq_idx];
    #ifndef USE_BMI2_SLIDERS
        bishop_magic_entries[sq_idx].magic_number = PRECOMPUTED_BISHOP_MAGICS[sq_idx];
        bishop_magic_entries[sq_idx].shift = PRECOMPUTED_BISHOP_SHIFTS[sq_idx];
    #endif

        std::vector<bitboard_t> permutations = get_blocker_permutations(bishop_magic_entries[sq_idx].mask);
        for (const auto& blockers : permutations) {
    #ifdef USE_BMI2_SLIDERS
            uint64_t pext_index = _pext_u64(blockers, bishop_magic_entries[sq_idx].mask);
            bishop_magic_entries[sq_idx].attacks[pext_index] = generate_attacks_slow_internal(sq_idx, blockers, false);
    #else
            uint64_t magic_index = (blockers * bishop_magic_entries[sq_idx].magic_number) >> bishop_magic_entries[sq_idx].shift;
            bishop_magic_entries[sq_idx].attacks[magic_index] = generate_attacks_slow_internal(sq_idx, blockers, false);
    #endif
        }
    }
}

// --- Slider Attack Lookup Functions ---

//--
/* get_rook_slider_attacks */
//--
// Retrieves precomputed rook attacks for a given square, considering current board occupancy.
// Uses magic bitboards for fast lookup.
// `sq`: The square from which the rook attacks.
// `occupied`: A bitboard representing all occupied squares on the board.
// Returns a bitboard of all squares attacked by a rook on `sq` with the given `occupied` state.
bitboard_t get_rook_slider_attacks(square_e sq, bitboard_t occupied) {
    const MagicEntry& entry = rook_magic_entries[static_cast<int>(sq)];
#ifdef USE_BMI2_SLIDERS
    // With PEXT, the index is directly calculated from occupied squares on the mask
    uint64_t pext_index = _pext_u64(occupied, entry.mask);
    return entry.attacks[pext_index];
#else
    // Original magic multiplication method
    bitboard_t blockers_on_mask = occupied & entry.mask;
    uint64_t index = (blockers_on_mask * entry.magic_number) >> entry.shift;
    return entry.attacks[index];
#endif
}

//--
/* get_bishop_slider_attacks */
//--
// Retrieves precomputed bishop attacks for a given square, considering current board occupancy.
// Uses magic bitboards for fast lookup.
// `sq`: The square from which the bishop attacks.
// `occupied`: A bitboard representing all occupied squares on the board.
// Returns a bitboard of all squares attacked by a bishop on `sq` with the given `occupied` state.
bitboard_t get_bishop_slider_attacks(square_e sq, bitboard_t occupied) {
    const MagicEntry& entry = bishop_magic_entries[static_cast<int>(sq)];
#ifdef USE_BMI2_SLIDERS
    uint64_t pext_index = _pext_u64(occupied, entry.mask);
    return entry.attacks[pext_index];
#else
    bitboard_t blockers_on_mask = occupied & entry.mask;
    uint64_t index = (blockers_on_mask * entry.magic_number) >> entry.shift;
    return entry.attacks[index];
#endif
}
} // namespace core   
} // namespace hyperion 