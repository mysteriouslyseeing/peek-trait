use core::slice;
use std::{
    collections::{btree_map, btree_set, hash_map, hash_set, linked_list, vec_deque},
    iter::{self, Peekable},
    ops, str, vec,
};

use delegate::delegate;

use crate::{Peek, PeekBack, PeekNth, PeekNthBack};

impl<I: Iterator> Peek for Peekable<I> {
    type PeekItem<'a>
        = &'a mut I::Item
    where
        Self: 'a;

    fn peek<'a>(&'a mut self) -> Option<Self::PeekItem<'a>> {
        self.peek_mut()
    }
}

impl<T> Peek for vec::IntoIter<T> {
    type PeekItem<'a>
        = &'a mut T
    where
        Self: 'a;

    fn peek<'a>(&'a mut self) -> Option<Self::PeekItem<'a>> {
        self.as_mut_slice().first_mut()
    }
}

impl<T> PeekNth for vec::IntoIter<T> {
    fn peek_nth<'a>(&'a mut self, n: usize) -> Option<Self::PeekItem<'a>> {
        self.as_mut_slice().get_mut(n)
    }
}

impl<T> PeekBack for vec::IntoIter<T> {
    fn peek_back<'a>(&'a mut self) -> Option<Self::PeekItem<'a>> {
        self.as_mut_slice().last_mut()
    }
}

impl<T> PeekNthBack for vec::IntoIter<T> {
    fn peek_nth_back<'a>(&'a mut self, n: usize) -> Option<Self::PeekItem<'a>> {
        let s = self.as_mut_slice();
        // `checked_sub` twice rather than `len - n - 1`, so `n == usize::MAX` can't overflow.
        let i = s.len().checked_sub(n)?.checked_sub(1)?;
        s.get_mut(i)
    }
}

impl<I: Peek + ?Sized> Peek for &mut I {
    type PeekItem<'a>
        = I::PeekItem<'a>
    where
        Self: 'a;

    delegate! {
        to (**self) {
            fn peek<'a>(&'a mut self) -> Option<Self::PeekItem<'a>>;
        }
    }
}
impl<I: PeekNth + ?Sized> PeekNth for &mut I {
    delegate! {
        to (**self) {
            fn peek_nth<'a>(&'a mut self, n: usize) -> Option<Self::PeekItem<'a>>;
        }
    }
}
impl<I: PeekBack + ?Sized> PeekBack for &mut I {
    delegate! {
        to (**self) {
            fn peek_back<'a>(&'a mut self) -> Option<Self::PeekItem<'a>>;
        }
    }
}
impl<I: PeekNthBack + ?Sized> PeekNthBack for &mut I {
    delegate! {
        to (**self) {
            fn peek_nth_back<'a>(&'a mut self, n: usize) -> Option<Self::PeekItem<'a>>;
        }
    }
}

impl<I: Peek + ?Sized> Peek for Box<I> {
    type PeekItem<'a>
        = I::PeekItem<'a>
    where
        Self: 'a;

    delegate! {
        to (**self) {
            fn peek<'a>(&'a mut self) -> Option<Self::PeekItem<'a>>;
        }
    }
}
impl<I: PeekNth + ?Sized> PeekNth for Box<I> {
    delegate! {
        to (**self) {
            fn peek_nth<'a>(&'a mut self, n: usize) -> Option<Self::PeekItem<'a>>;
        }
    }
}
impl<I: PeekBack + ?Sized> PeekBack for Box<I> {
    delegate! {
        to (**self) {
            fn peek_back<'a>(&'a mut self) -> Option<Self::PeekItem<'a>>;
        }
    }
}
impl<I: PeekNthBack + ?Sized> PeekNthBack for Box<I> {
    delegate! {
        to (**self) {
            fn peek_nth_back<'a>(&'a mut self, n: usize) -> Option<Self::PeekItem<'a>>;
        }
    }
}

macro_rules! clone_peek {
    (frwd [$($g:tt)*] $ty:ty $(where [$($w:tt)*])?) => {
        clone_peek!(@front [$($g)*] $ty, [$($($w)*)?]);
    };
    (both [$($g:tt)*] $ty:ty $(where [$($w:tt)*])?) => {
        clone_peek!(@front [$($g)*] $ty, [$($($w)*)?]);
        clone_peek!(@back [$($g)*] $ty, [$($($w)*)?]);
    };

    (@front [$($g:tt)*] $ty:ty, [$($w:tt)*]) => {
        impl<$($g)*> Peek for $ty
        where
            $ty: Iterator + Clone,
            $($w)*
        {
            type PeekItem<'peek>
                = <$ty as Iterator>::Item
            where
                Self: 'peek;

            fn peek<'peek>(&'peek mut self) -> Option<Self::PeekItem<'peek>> {
                self.clone().next()
            }
        }

        impl<$($g)*> PeekNth for $ty
        where
            $ty: Iterator + Clone,
            $($w)*
        {
            fn peek_nth<'peek>(&'peek mut self, n: usize) -> Option<Self::PeekItem<'peek>> {
                self.clone().nth(n)
            }
        }
    };

    (@back [$($g:tt)*] $ty:ty, [$($w:tt)*]) => {
        impl<$($g)*> PeekBack for $ty
        where
            $ty: DoubleEndedIterator + Clone,
            $($w)*
        {
            fn peek_back<'peek>(&'peek mut self) -> Option<Self::PeekItem<'peek>> {
                self.clone().next_back()
            }
        }

        impl<$($g)*> PeekNthBack for $ty
        where
            $ty: DoubleEndedIterator + Clone,
            $($w)*
        {
            fn peek_nth_back<'peek>(&'peek mut self, n: usize) -> Option<Self::PeekItem<'peek>> {
                self.clone().nth_back(n)
            }
        }
    };
}

// Ranges. `Range<A>: Iterator` already encodes the unstable `Step` bound, so none is named.
clone_peek!(both [A] ops::Range<A>);
clone_peek!(both [A] ops::RangeInclusive<A>);
clone_peek!(frwd [A] ops::RangeFrom<A>);

// Strings. O(1) to clone, but the `nth` variants are O(n).
clone_peek!(both ['s] str::Chars<'s>);
clone_peek!(both ['s] str::CharIndices<'s>);
clone_peek!(both ['s] str::Bytes<'s>);

// Slice adaptors.
clone_peek!(both ['s, T] slice::Windows<'s, T>);
clone_peek!(both ['s, T] slice::Chunks<'s, T>);
clone_peek!(both ['s, T] slice::ChunksExact<'s, T>);
clone_peek!(both ['s, T] slice::RChunks<'s, T>);

// Collections.
clone_peek!(both ['s, T] vec_deque::Iter<'s, T>);
clone_peek!(both ['s, K, V] btree_map::Iter<'s, K, V>);
clone_peek!(both ['s, K, V] btree_map::Keys<'s, K, V>);
clone_peek!(both ['s, K, V] btree_map::Values<'s, K, V>);
clone_peek!(both ['s, T] btree_set::Iter<'s, T>);
clone_peek!(frwd ['s, K, V] hash_map::Iter<'s, K, V>);
clone_peek!(frwd ['s, K, V] hash_map::Keys<'s, K, V>);
clone_peek!(frwd ['s, K, V] hash_map::Values<'s, K, V>);
clone_peek!(frwd ['s, T] hash_set::Iter<'s, T>);
clone_peek!(both ['s, T] linked_list::Iter<'s, T>);

// Adaptors over a cloneable inner iterator. The macro's `Self: Iterator + Clone` bound is
// already enough for the inner bounds to follow, so no `where` is needed.
clone_peek!(both [I] iter::Rev<I>);
clone_peek!(both [I] iter::Enumerate<I>);
clone_peek!(both [I] iter::Skip<I>);
clone_peek!(both [I] iter::Take<I>);
clone_peek!(both [I] iter::StepBy<I>);
clone_peek!(both [I] iter::Fuse<I>);
clone_peek!(both [I] iter::Cloned<I>);
clone_peek!(both [I] iter::Copied<I>);
clone_peek!(both [A, B] iter::Zip<A, B>);
clone_peek!(both [A, B] iter::Chain<A, B>);
clone_peek!(frwd [I] iter::Cycle<I>);
