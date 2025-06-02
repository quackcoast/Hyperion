// hyperion/src/cpp/core/bitboard.cpp
#include "bitboard.hpp"
#include "constants.hpp"
#include "position.hpp"

#include <algorithm>
#include <vector>

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

namespace { // Anonymous namespace for internal linkage

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
        std::vector<int> set_bits_indices;
        for(int i=0; i<NUM_SQUARES; ++i) {
            if(get_bit(mask, i)) {
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
        return permutations;
    }
} // end anonymous namespace

//---------------------------//
//                           //
// PRECOMPUTED ATTACKS BELOW //
//                           //
//---------------------------//

namespace {
        // --- ROOK MAGICS ---
    const uint64_t PRECOMPUTED_ROOK_MAGICS[64] = {
        0x10600a0202010081ULL, // sq 0
        0x0022041020404010ULL, // sq 1
        0x2402001102100008ULL, // sq 2
        0x1200241001010018ULL, // sq 3
        0x0400221001010008ULL, // sq 4
        0x0400200804080401ULL, // sq 5
        0x4026000401002093ULL, // sq 6
        0x0001208201000145ULL, // sq 7
        0x0180284320440402ULL, // sq 8
        0x5400420c40004020ULL, // sq 9
        0x086d300024402008ULL, // sq 10
        0x0009404060804410ULL, // sq 11
        0x03e0401009204006ULL, // sq 12
        0x0180601200104042ULL, // sq 13
        0x0004010800020002ULL, // sq 14
        0x0320024008102001ULL, // sq 15
        0x0220128000004040ULL, // sq 16
        0x0800550400412020ULL, // sq 17
        0x00204a2010001820ULL, // sq 18
        0x0009088008100104ULL, // sq 19
        0x3485004020000808ULL, // sq 20
        0x3014008800200202ULL, // sq 21
        0x0041080400a80202ULL, // sq 22
        0x00500a8018000101ULL, // sq 23
        0x038468c040020102ULL, // sq 24
        0x0095102002021442ULL, // sq 25
        0x442001a020044415ULL, // sq 26
        0x4023081000064202ULL, // sq 27
        0x2100080004001145ULL, // sq 28
        0x0004100a20000202ULL, // sq 29
        0x0004080110045614ULL, // sq 30
        0x20a0440120000092ULL, // sq 31
        0x0080400604002089ULL, // sq 32
        0x0010210a09c00804ULL, // sq 33
        0x0100201018a07005ULL, // sq 34
        0x0110002000141801ULL, // sq 35
        0x0c0800400004011aULL, // sq 36
        0x088004820008408eULL, // sq 37
        0x0a00189c42801182ULL, // sq 38
        0x0040008001200449ULL, // sq 39
        0x10404010228200a0ULL, // sq 40
        0x014840010c002210ULL, // sq 41
        0x0109006002001020ULL, // sq 42
        0x1400404440244008ULL, // sq 43
        0x0280206004210806ULL, // sq 44
        0x2140400440010002ULL, // sq 45
        0x1242020000204409ULL, // sq 46
        0x0835008004080087ULL, // sq 47
        0x4012204004006001ULL, // sq 48
        0x02a1404000404418ULL, // sq 49
        0x1020026002900408ULL, // sq 50
        0x12506b004c100049ULL, // sq 51
        0x0020440440000489ULL, // sq 52
        0x00c0202000220085ULL, // sq 53
        0x0040041040001006ULL, // sq 54
        0x08800040000000c1ULL, // sq 55
        0x2420144160050080ULL, // sq 56
        0x04c1028211092012ULL, // sq 57
        0x0280300a08322842ULL, // sq 58
        0x004500c904002010ULL, // sq 59
        0x0c44104300010008ULL, // sq 60
        0x4100082508000204ULL, // sq 61
        0x0300008100800402ULL, // sq 62
        0x002040af10410021ULL, // sq 63
    };

    const bitboard_t PRECOMPUTED_ROOK_MASKS[64] = {
        0x0000000000000000ULL,
        0x0002020202020200ULL,
        0x0004040404040400ULL,
        0x0008080808080800ULL,
        0x0010101010101000ULL,
        0x0020202020202000ULL,
        0x0040404040404000ULL,
        0x0000000000000000ULL,
        0x0000000000007e00ULL,
        0x0002020202027c00ULL,
        0x0004040404047a00ULL,
        0x0008080808087600ULL,
        0x0010101010106e00ULL,
        0x0020202020205e00ULL,
        0x0040404040403e00ULL,
        0x0000000000007e00ULL,
        0x00000000007e0000ULL,
        0x00020202027c0200ULL,
        0x00040404047a0400ULL,
        0x0008080808760800ULL,
        0x00101010106e1000ULL,
        0x00202020205e2000ULL,
        0x00404040403e4000ULL,
        0x00000000007e0000ULL,
        0x000000007e000000ULL,
        0x000202027c020200ULL,
        0x000404047a040400ULL,
        0x0008080876080800ULL,
        0x001010106e101000ULL,
        0x002020205e202000ULL,
        0x004040403e404000ULL,
        0x000000007e000000ULL,
        0x0000007e00000000ULL,
        0x0002027c02020200ULL,
        0x0004047a04040400ULL,
        0x0008087608080800ULL,
        0x0010106e10101000ULL,
        0x0020205e20202000ULL,
        0x0040403e40404000ULL,
        0x0000007e00000000ULL,
        0x00007e0000000000ULL,
        0x00027c0202020200ULL,
        0x00047a0404040400ULL,
        0x0008760808080800ULL,
        0x00106e1010101000ULL,
        0x00205e2020202000ULL,
        0x00403e4040404000ULL,
        0x00007e0000000000ULL,
        0x007e000000000000ULL,
        0x007c020202020200ULL,
        0x007a040404040400ULL,
        0x0076080808080800ULL,
        0x006e101010101000ULL,
        0x005e202020202000ULL,
        0x003e404040404000ULL,
        0x007e000000000000ULL,
        0x0000000000000000ULL,
        0x0002020202020200ULL,
        0x0004040404040400ULL,
        0x0008080808080800ULL,
        0x0010101010101000ULL,
        0x0020202020202000ULL,
        0x0040404040404000ULL,
        0x0000000000000000ULL
    };

    const uint8_t PRECOMPUTED_ROOK_SHIFTS[64] = {
        64,
        58,
        58,
        58,
        58,
        58,
        58,
        64,
        58,
        54,
        54,
        54,
        54,
        54,
        54,
        58,
        58,
        54,
        54,
        54,
        54,
        54,
        54,
        58,
        58,
        54,
        54,
        54,
        54,
        54,
        54,
        58,
        58,
        54,
        54,
        54,
        54,
        54,
        54,
        58,
        58,
        54,
        54,
        54,
        54,
        54,
        54,
        58,
        58,
        54,
        54,
        54,
        54,
        54,
        54,
        58,
        64,
        58,
        58,
        58,
        58,
        58,
        58,
        64
    };

    const uint32_t PRECOMPUTED_ROOK_OFFSETS[64] = {
        0,
        1,
        65,
        129,
        193,
        257,
        321,
        385,
        386,
        450,
        1474,
        2498,
        3522,
        4546,
        5570,
        6594,
        6658,
        6722,
        7746,
        8770,
        9794,
        10818,
        11842,
        12866,
        12930,
        12994,
        14018,
        15042,
        16066,
        17090,
        18114,
        19138,
        19202,
        19266,
        20290,
        21314,
        22338,
        23362,
        24386,
        25410,
        25474,
        25538,
        26562,
        27586,
        28610,
        29634,
        30658,
        31682,
        31746,
        31810,
        32834,
        33858,
        34882,
        35906,
        36930,
        37954,
        38018,
        38019,
        38083,
        38147,
        38211,
        38275,
        38339,
        38403
    };


    // --- BISHOP MAGICS ---
const uint64_t PRECOMPUTED_BISHOP_MAGICS[64] = {
        0x0401021002880125ULL, // sq 0
        0x0200330000951102ULL, // sq 1
        0x04f8240a26121404ULL, // sq 2
        0x0005229000040502ULL, // sq 3
        0x0022280000020210ULL, // sq 4
        0x2806220040060184ULL, // sq 5
        0x2240186001290c02ULL, // sq 6
        0x12a4080210812104ULL, // sq 7
        0x1842008414822002ULL, // sq 8
        0x0482008001862002ULL, // sq 9
        0x0491211004881004ULL, // sq 10
        0x0480200a00a02c5cULL, // sq 11
        0x0800000520c01443ULL, // sq 12
        0x2012420904800101ULL, // sq 13
        0x102a248005010129ULL, // sq 14
        0x1082101115000201ULL, // sq 15
        0x2208452302110004ULL, // sq 16
        0x141204a1214a2011ULL, // sq 17
        0x2059024000081001ULL, // sq 18
        0x4020180310020004ULL, // sq 19
        0x0422008007081214ULL, // sq 20
        0x08a0140800082001ULL, // sq 21
        0x0a04456200204c02ULL, // sq 22
        0x0148440000405821ULL, // sq 23
        0x08020c0044144080ULL, // sq 24
        0x20110400200a0830ULL, // sq 25
        0x040c402008002200ULL, // sq 26
        0x0240150000004120ULL, // sq 27
        0x0090410004414100ULL, // sq 28
        0x0620512000500040ULL, // sq 29
        0x040104020804014dULL, // sq 30
        0x0042020100084323ULL, // sq 31
        0x0141082002225004ULL, // sq 32
        0x001801022008090cULL, // sq 33
        0x4020428601003110ULL, // sq 34
        0x400022804008410aULL, // sq 35
        0x3004110003c00047ULL, // sq 36
        0x0211208000040882ULL, // sq 37
        0x022241101f460084ULL, // sq 38
        0x0408328002480109ULL, // sq 39
        0x2081088821021002ULL, // sq 40
        0x1404488111040082ULL, // sq 41
        0x1010010060804066ULL, // sq 42
        0x0422220800235202ULL, // sq 43
        0x00c0040031824105ULL, // sq 44
        0x4041220410285081ULL, // sq 45
        0x4204411820200301ULL, // sq 46
        0x3820388002520484ULL, // sq 47
        0x4a0800022c162090ULL, // sq 48
        0x0602148002b24104ULL, // sq 49
        0x0441104100410441ULL, // sq 50
        0x6292020005040002ULL, // sq 51
        0x3201041008360008ULL, // sq 52
        0x0802500009400430ULL, // sq 53
        0x45820c4020040810ULL, // sq 54
        0x0040410001a205a4ULL, // sq 55
        0x7a82100104016218ULL, // sq 56
        0x0488040a08800a01ULL, // sq 57
        0x00c1101300d62011ULL, // sq 58
        0x12c20600000404b0ULL, // sq 59
        0x5222060004000c11ULL, // sq 60
        0x0c08608800900140ULL, // sq 61
        0x4102008000806620ULL, // sq 62
        0x0126009060c00644ULL, // sq 63
    };

    const bitboard_t PRECOMPUTED_BISHOP_MASKS[64] = {
        0x0040201008040200ULL,
        0x0000402010080400ULL,
        0x0000004020100a00ULL,
        0x0000000040221400ULL,
        0x0000000002442800ULL,
        0x0000000204085000ULL,
        0x0000020408102000ULL,
        0x0002040810204000ULL,
        0x0020100804020000ULL,
        0x0040201008040000ULL,
        0x00004020100a0000ULL,
        0x0000004022140000ULL,
        0x0000000244280000ULL,
        0x0000020408500000ULL,
        0x0002040810200000ULL,
        0x0004081020400000ULL,
        0x0010080402000200ULL,
        0x0020100804000400ULL,
        0x004020100a000a00ULL,
        0x0000402214001400ULL,
        0x0000024428002800ULL,
        0x0002040850005000ULL,
        0x0004081020002000ULL,
        0x0008102040004000ULL,
        0x0008040200020400ULL,
        0x0010080400040800ULL,
        0x0020100a000a1000ULL,
        0x0040221400142200ULL,
        0x0002442800284400ULL,
        0x0004085000500800ULL,
        0x0008102000201000ULL,
        0x0010204000402000ULL,
        0x0004020002040800ULL,
        0x0008040004081000ULL,
        0x00100a000a102000ULL,
        0x0022140014224000ULL,
        0x0044280028440200ULL,
        0x0008500050080400ULL,
        0x0010200020100800ULL,
        0x0020400040201000ULL,
        0x0002000204081000ULL,
        0x0004000408102000ULL,
        0x000a000a10204000ULL,
        0x0014001422400000ULL,
        0x0028002844020000ULL,
        0x0050005008040200ULL,
        0x0020002010080400ULL,
        0x0040004020100800ULL,
        0x0000020408102000ULL,
        0x0000040810204000ULL,
        0x00000a1020400000ULL,
        0x0000142240000000ULL,
        0x0000284402000000ULL,
        0x0000500804020000ULL,
        0x0000201008040200ULL,
        0x0000402010080400ULL,
        0x0002040810204000ULL,
        0x0004081020400000ULL,
        0x000a102040000000ULL,
        0x0014224000000000ULL,
        0x0028440200000000ULL,
        0x0050080402000000ULL,
        0x0020100804020000ULL,
        0x0040201008040200ULL
    };

    const uint8_t PRECOMPUTED_BISHOP_SHIFTS[64] = {
        58,
        59,
        59,
        59,
        59,
        59,
        59,
        58,
        59,
        59,
        59,
        59,
        59,
        59,
        59,
        59,
        59,
        59,
        57,
        57,
        57,
        57,
        59,
        59,
        59,
        59,
        57,
        55,
        55,
        57,
        59,
        59,
        59,
        59,
        57,
        55,
        55,
        57,
        59,
        59,
        59,
        59,
        57,
        57,
        57,
        57,
        59,
        59,
        59,
        59,
        59,
        59,
        59,
        59,
        59,
        59,
        58,
        59,
        59,
        59,
        59,
        59,
        59,
        58
    };

    // Total Bishop Attack Table Entries: 5248
    const uint32_t PRECOMPUTED_BISHOP_OFFSETS[64] = {
        0,
        64,
        96,
        128,
        160,
        192,
        224,
        256,
        320,
        352,
        384,
        416,
        448,
        480,
        512,
        544,
        576,
        608,
        640,
        768,
        896,
        1024,
        1152,
        1184,
        1216,
        1248,
        1280,
        1408,
        1920,
        2432,
        2560,
        2592,
        2624,
        2656,
        2688,
        2816,
        3328,
        3840,
        3968,
        4000,
        4032,
        4064,
        4096,
        4224,
        4352,
        4480,
        4608,
        4640,
        4672,
        4704,
        4736,
        4768,
        4800,
        4832,
        4864,
        4896,
        4928,
        4992,
        5024,
        5056,
        5088,
        5120,
        5152,
        5184
    };
} //end anonymous namespace

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
        rook_magic_entries[sq_idx].magic_number = PRECOMPUTED_ROOK_MAGICS[sq_idx];
        rook_magic_entries[sq_idx].mask = PRECOMPUTED_ROOK_MASKS[sq_idx];
        rook_magic_entries[sq_idx].shift = PRECOMPUTED_ROOK_SHIFTS[sq_idx];
        rook_magic_entries[sq_idx].attacks = rook_attack_table + PRECOMPUTED_ROOK_OFFSETS[sq_idx];

        std::vector<bitboard_t> permutations = get_blocker_permutations(rook_magic_entries[sq_idx].mask);
        for (const auto& blockers : permutations) {
            uint64_t magic_index = (blockers * rook_magic_entries[sq_idx].magic_number) >> rook_magic_entries[sq_idx].shift;
            rook_magic_entries[sq_idx].attacks[magic_index] = generate_attacks_slow_internal(sq_idx, blockers, true);
        }
    }

    // Initialize Bishop Magic Entries and Bishop Attack Table
    for (int sq_idx = 0; sq_idx < NUM_SQUARES; ++sq_idx) { //
        bishop_magic_entries[sq_idx].magic_number = PRECOMPUTED_BISHOP_MAGICS[sq_idx];
        bishop_magic_entries[sq_idx].mask = PRECOMPUTED_BISHOP_MASKS[sq_idx]; 
        bishop_magic_entries[sq_idx].shift = PRECOMPUTED_BISHOP_SHIFTS[sq_idx];
        bishop_magic_entries[sq_idx].attacks = bishop_attack_table + PRECOMPUTED_BISHOP_OFFSETS[sq_idx];

        std::vector<bitboard_t> permutations = get_blocker_permutations(bishop_magic_entries[sq_idx].mask);
        for (const auto& blockers : permutations) {
            uint64_t magic_index = (blockers * bishop_magic_entries[sq_idx].magic_number) >> bishop_magic_entries[sq_idx].shift;
            bishop_magic_entries[sq_idx].attacks[magic_index] = generate_attacks_slow_internal(sq_idx, blockers, false);
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
    bitboard_t blockers = occupied & entry.mask;
    uint64_t index = (blockers * entry.magic_number) >> entry.shift;
    return entry.attacks[index];
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
    bitboard_t blockers = occupied & entry.mask;
    uint64_t index = (blockers * entry.magic_number) >> entry.shift;
    return entry.attacks[index];
}

} // namespace core   
} // namespace hyperion 