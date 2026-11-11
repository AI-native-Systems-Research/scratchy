// SPDX-License-Identifier: Apache-2.0
//! ⭐ THE CARDS THIS HOST HAS, READ ONCE — senlib's enumeration as one value.
//!
//! senlib numbers the cards it enumerates `0..ncards` and names each by its PCIe address
//! (`SenPci::pf(i)`). Everything above this crate that picks cards picks them FROM this table, so
//! the enumeration order is stated in one place — here — instead of being re-derived as an integer
//! at every site that opens a card.
//!
//! ⛔ NO ENVIRONMENT CHOOSES CARDS HERE. The vendor's `SPYRE_DEVICES` / `AIU_WORLD_RANK_*` mapping
//! ([`crate::flex_config::FlexConfig::flex_spyre_devices`]) is not read: the table is what senlib
//! enumerates, in its order.

use crate::senlib_ffi::{pcie_address_for_card, sen_pci_card_count};

/// One enumerated card: its index in senlib's enumeration, and its PCIe address.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EnumeratedCard {
    index: u32,
    pci: String,
}

impl EnumeratedCard {
    /// Its index in senlib's enumeration.
    pub fn index(&self) -> u32 {
        self.index
    }

    /// Its PCIe address — what `Card::open` opens it by.
    pub fn pci(&self) -> &str {
        &self.pci
    }
}

/// Every card senlib enumerates on this host, in its order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CardTable {
    cards: Vec<EnumeratedCard>,
}

/// Why the table could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CardTableError {
    /// senlib reported a negative card count.
    BadCount(i64),
    /// senlib enumerates a card at `index` but names no PCIe address for it.
    NoAddress { index: u32 },
}

impl std::fmt::Display for CardTableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadCount(n) => write!(f, "senlib reports {n} Spyre cards"),
            Self::NoAddress { index } => {
                write!(
                    f,
                    "senlib enumerates Spyre card {index} but names no PCIe address for it"
                )
            }
        }
    }
}

impl std::error::Error for CardTableError {}

impl CardTable {
    /// Read senlib's enumeration: its count, then each card's PCIe address, once.
    ///
    /// # Errors
    /// [`CardTableError`], by name.
    pub fn enumerate() -> Result<CardTable, CardTableError> {
        let n = sen_pci_card_count();
        let n = u32::try_from(n).map_err(|_| CardTableError::BadCount(n))?;
        let cards = (0..n)
            .map(|index| {
                let pci = pcie_address_for_card(index);
                if pci.is_empty() {
                    Err(CardTableError::NoAddress { index })
                } else {
                    Ok(EnumeratedCard { index, pci })
                }
            })
            .collect::<Result<_, _>>()?;
        Ok(CardTable { cards })
    }

    /// A table of these cards, in this order — what a test stands in for senlib with.
    pub fn of(cards: impl IntoIterator<Item = (u32, String)>) -> CardTable {
        CardTable {
            cards: cards
                .into_iter()
                .map(|(index, pci)| EnumeratedCard { index, pci })
                .collect(),
        }
    }

    /// The cards, in senlib's order.
    pub fn cards(&self) -> &[EnumeratedCard] {
        &self.cards
    }
}
