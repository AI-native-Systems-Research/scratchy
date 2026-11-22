// SPDX-License-Identifier: Apache-2.0
//! A tensor value as KTIR states it: a strided view over a buffer's elements.
//!
//! `ktdp.construct_memory_view` states a buffer's shape and strides, an access tile takes a window
//! of it, and every shape op (`expand_shape`, `collapse_shape`, `reshape`, `transpose`,
//! `broadcast`) re-describes the same elements. A [`View`] is the composition of all of them: the
//! element offset of logical index `i` is `offset + Σ i[a]·stride[a]`. That is the KTIR semantics
//! verbatim, so the bytes a lowered descriptor addresses are the bytes the emulator reads.

/// One axis of a view: `extent` positions, `stride` elements apart. A stride of 0 repeats one
/// element along the axis — a broadcast.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Axis {
    pub extent: u64,
    pub stride: u64,
}

/// A strided view over a buffer's elements, axes outermost first (KTIR's logical order).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct View {
    /// Element offset of the view's origin from the buffer's origin.
    pub offset: u64,
    pub axes: Vec<Axis>,
    /// `Some` when one unit-stride axis is held in stick columns: its sticks are `pitch` elements
    /// apart rather than consecutive.
    pub tile: Option<Tile>,
}

/// A stick-column axis: `axis` (stride 1) holds `lanes` elements per stick, and its consecutive
/// sticks are `pitch` elements apart — the device's KERNEL layout `[n/lanes, k, lanes]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tile {
    pub axis: usize,
    pub lanes: u64,
    pub pitch: u64,
}

impl Tile {
    /// Element distance of position `c` along the tiled axis from its origin.
    pub const fn at(self, c: u64) -> u64 {
        (c / self.lanes) * self.pitch + c % self.lanes
    }
}

/// A shape op the view cannot express without moving elements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ViewError {
    /// A reshape whose merged axes are not contiguous in the buffer.
    NotContiguous { from: Vec<u64>, to: Vec<u64> },
    /// The element counts before and after a reshape differ.
    Numel { from: u64, to: u64 },
    /// A permutation that is not one of the view's rank.
    Permutation { rank: usize, perm: Vec<usize> },
    /// A window that reads past an axis.
    Window {
        axis: usize,
        corner: u64,
        extent: u64,
        of: u64,
    },
    /// A reshape of a view held in stick columns, or a window that starts inside a stick column.
    Tiled,
}

impl View {
    /// `ktdp.construct_memory_view`: the buffer's own shape and strides, at its origin.
    pub fn of_memory(shape: &[u64], strides: &[u64]) -> View {
        View {
            offset: 0,
            axes: shape
                .iter()
                .zip(strides)
                .map(|(&extent, &stride)| Axis { extent, stride })
                .collect(),
            tile: None,
        }
    }

    /// A row-major view of `shape` at element `offset`.
    pub fn row_major(offset: u64, shape: &[u64]) -> View {
        let mut stride = 1;
        let mut axes: Vec<Axis> = shape
            .iter()
            .rev()
            .map(|&extent| {
                let a = Axis { extent, stride };
                stride *= extent;
                a
            })
            .collect();
        axes.reverse();
        View {
            offset,
            axes,
            tile: None,
        }
    }

    /// The logical shape.
    pub fn shape(&self) -> Vec<u64> {
        self.axes.iter().map(|a| a.extent).collect()
    }

    /// Whether the two views name the same elements in the same order: an axis of one position
    /// names nothing, wherever it sits and whatever its stride.
    pub fn same_elements(&self, other: &View) -> bool {
        let spread =
            |v: &View| -> Vec<Axis> { v.axes.iter().copied().filter(|a| a.extent > 1).collect() };
        self.offset == other.offset
            && match (self.tile, other.tile) {
                (None, None) => spread(self) == spread(other),
                _ => self.tile == other.tile && self.axes == other.axes,
            }
    }

    /// This view over a space of `to` — `to` and the view's own shape the same axes apart from
    /// axes of one position — with an axis of one position wherever `to` has one. `None` when the
    /// two spaces differ in more than such axes, or the view is held in stick columns.
    pub fn over_space(&self, to: &[u64]) -> Option<View> {
        let spread = |s: &[u64]| -> Vec<u64> { s.iter().copied().filter(|&e| e > 1).collect() };
        if self.tile.is_some() || spread(&self.shape()) != spread(to) {
            return None;
        }
        let mut own = self.axes.iter().filter(|a| a.extent > 1);
        let axes = to
            .iter()
            .map(|&e| match e {
                1 => Some(Axis {
                    extent: 1,
                    stride: 0,
                }),
                _ => own.next().copied(),
            })
            .collect::<Option<Vec<Axis>>>()?;
        Some(View {
            offset: self.offset,
            axes,
            tile: None,
        })
    }

    /// Element count.
    pub fn numel(&self) -> u64 {
        self.axes.iter().map(|a| a.extent).product()
    }

    /// One past the furthest element the view reaches, from the buffer origin.
    pub fn end(&self) -> u64 {
        self.offset
            + self
                .axes
                .iter()
                .enumerate()
                .map(|(i, a)| self.distance(i, a.extent.saturating_sub(1)))
                .sum::<u64>()
            + 1
    }

    /// Element distance of position `c` along axis `i` from the axis's origin.
    pub fn distance(&self, i: usize, c: u64) -> u64 {
        match self.tile {
            Some(t) if t.axis == i => t.at(c),
            _ => c * self.axes[i].stride,
        }
    }

    /// An access tile: the `shape` window whose origin is `corner`.
    pub fn window(&self, corner: &[u64], shape: &[u64]) -> Result<View, ViewError> {
        let mut offset = self.offset;
        let mut axes = Vec::with_capacity(self.axes.len());
        for (i, ((a, &c), &e)) in self.axes.iter().zip(corner).zip(shape).enumerate() {
            if c + e > a.extent {
                return Err(ViewError::Window {
                    axis: i,
                    corner: c,
                    extent: e,
                    of: a.extent,
                });
            }
            if self.tile.is_some_and(|t| t.axis == i && c % t.lanes != 0) {
                return Err(ViewError::Tiled);
            }
            offset += self.distance(i, c);
            axes.push(Axis {
                extent: e,
                stride: a.stride,
            });
        }
        Ok(View {
            offset,
            axes,
            tile: self.tile,
        })
    }

    /// Advance the origin by `corner` along each axis, keeping the extents — an access tile whose
    /// shape is the view's own.
    pub fn shifted(&self, corner: &[u64]) -> View {
        let offset = self.offset
            + corner
                .iter()
                .enumerate()
                .map(|(i, &c)| self.distance(i, c))
                .sum::<u64>();
        View {
            offset,
            axes: self.axes.clone(),
            tile: self.tile,
        }
    }

    /// `linalg.transpose` / a permuted read: result axis `i` is source axis `perm[i]`.
    pub fn permute(&self, perm: &[usize]) -> Result<View, ViewError> {
        let mut seen = vec![false; self.axes.len()];
        if perm.len() != self.axes.len()
            || perm
                .iter()
                .any(|&p| p >= seen.len() || std::mem::replace(&mut seen[p], true))
        {
            return Err(ViewError::Permutation {
                rank: self.axes.len(),
                perm: perm.to_vec(),
            });
        }
        Ok(View {
            offset: self.offset,
            axes: perm.iter().map(|&p| self.axes[p]).collect(),
            tile: self.tile.map(|t| Tile {
                axis: perm.iter().position(|&p| p == t.axis).unwrap_or(t.axis),
                ..t
            }),
        })
    }

    /// `linalg.broadcast`: the result has rank `self.rank() + added.len()`; the result axes listed
    /// in `added` repeat, the rest are this view's axes in order.
    pub fn broadcast(&self, out_shape: &[u64], added: &[usize]) -> View {
        let mut src = self.axes.iter();
        let kept: Vec<usize> = (0..out_shape.len())
            .filter(|i| !added.contains(i))
            .collect();
        let axes = out_shape
            .iter()
            .enumerate()
            .map(|(i, &extent)| {
                if added.contains(&i) {
                    Axis { extent, stride: 0 }
                } else {
                    let a = src.next().copied().unwrap_or(Axis { extent, stride: 0 });
                    Axis {
                        extent,
                        stride: if a.extent == 1 && extent != 1 {
                            0
                        } else {
                            a.stride
                        },
                    }
                }
            })
            .collect();
        View {
            offset: self.offset,
            axes,
            tile: self.tile.map(|t| Tile {
                axis: kept.get(t.axis).copied().unwrap_or(t.axis),
                ..t
            }),
        }
    }

    /// Repeat size-1 axes out to `out_shape` (an implicit numpy broadcast of equal rank).
    pub fn broadcast_to(&self, out_shape: &[u64]) -> View {
        View {
            offset: self.offset,
            axes: self
                .axes
                .iter()
                .zip(out_shape)
                .map(|(a, &extent)| Axis {
                    extent,
                    stride: if a.extent == 1 && extent != 1 {
                        0
                    } else {
                        a.stride
                    },
                })
                .collect(),
            tile: self.tile,
        }
    }

    /// A row-major reshape (`expand_shape`, `collapse_shape`, `reshape`) without moving elements:
    /// numpy's no-copy reshape. Axes of extent 1 carry no stride; a group of source axes merges
    /// only where each is contiguous inside the next.
    pub fn reshape(&self, to: &[u64]) -> Result<View, ViewError> {
        if self.tile.is_some() {
            return Err(ViewError::Tiled);
        }
        let from_n = self.numel();
        let to_n: u64 = to.iter().product();
        if from_n != to_n {
            return Err(ViewError::Numel {
                from: from_n,
                to: to_n,
            });
        }
        let src: Vec<Axis> = self
            .axes
            .iter()
            .copied()
            .filter(|a| a.extent != 1)
            .collect();
        let dst: Vec<(usize, u64)> = to
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, e)| e != 1)
            .collect();
        let mut out = vec![
            Axis {
                extent: 1,
                stride: 0
            };
            to.len()
        ];
        for (i, &e) in to.iter().enumerate() {
            out[i].extent = e;
        }
        let err = || ViewError::NotContiguous {
            from: self.shape(),
            to: to.to_vec(),
        };
        let (mut si, mut di) = (0usize, 0usize);
        while si < src.len() && di < dst.len() {
            // Grow a source group and a destination group until their element counts agree.
            let (s0, d0) = (si, di);
            let mut sn = src[si].extent;
            let mut dn = dst[di].1;
            si += 1;
            di += 1;
            while sn != dn {
                if sn < dn {
                    let Some(a) = src.get(si) else {
                        return Err(err());
                    };
                    sn *= a.extent;
                    si += 1;
                } else {
                    let Some(&(_, e)) = dst.get(di) else {
                        return Err(err());
                    };
                    dn *= e;
                    di += 1;
                }
            }
            // The source group must be one contiguous run: each axis's stride is the next one's
            // stride times its extent (a broadcast group stays a broadcast).
            for k in s0..si - 1 {
                if src[k].stride != src[k + 1].stride * src[k + 1].extent {
                    return Err(err());
                }
            }
            let mut stride = src[si - 1].stride;
            for k in (d0..di).rev() {
                let (pos, e) = dst[k];
                out[pos].stride = stride;
                stride *= e;
            }
        }
        Ok(View {
            offset: self.offset,
            axes: out,
            tile: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_row_major_split_and_merge_round_trips() {
        let v = View::row_major(0, &[1, 4096]);
        let r = v.reshape(&[8, 4, 128]).unwrap();
        assert_eq!(
            r.axes,
            vec![
                Axis {
                    extent: 8,
                    stride: 512
                },
                Axis {
                    extent: 4,
                    stride: 128
                },
                Axis {
                    extent: 128,
                    stride: 1
                }
            ]
        );
        assert_eq!(
            r.reshape(&[1, 4096]).unwrap().axes[1],
            Axis {
                extent: 4096,
                stride: 1
            }
        );
    }

    #[test]
    fn a_window_merges_only_where_contiguous() {
        let v = View::row_major(0, &[64, 4096])
            .window(&[0, 1024], &[64, 1024])
            .unwrap();
        assert_eq!(v.offset, 1024);
        assert!(v.reshape(&[64 * 1024]).is_err());
        assert_eq!(
            v.reshape(&[64, 16, 64]).unwrap().axes,
            vec![
                Axis {
                    extent: 64,
                    stride: 4096
                },
                Axis {
                    extent: 16,
                    stride: 64
                },
                Axis {
                    extent: 64,
                    stride: 1
                }
            ]
        );
    }

    #[test]
    fn a_broadcast_repeats_its_added_axes() {
        let w = View::row_major(0, &[4096]);
        let b = w.broadcast(&[64, 4096], &[0]);
        assert_eq!(
            b.axes[0],
            Axis {
                extent: 64,
                stride: 0
            }
        );
        assert_eq!(
            b.axes[1],
            Axis {
                extent: 4096,
                stride: 1
            }
        );
    }

    #[test]
    fn a_transpose_permutes_the_strides() {
        let w = View::row_major(0, &[2048, 512]).permute(&[1, 0]).unwrap();
        assert_eq!(
            w.axes[0],
            Axis {
                extent: 512,
                stride: 1
            }
        );
        assert_eq!(
            w.axes[1],
            Axis {
                extent: 2048,
                stride: 512
            }
        );
    }
}
